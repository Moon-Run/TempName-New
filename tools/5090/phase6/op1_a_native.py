"""Isolated OP1 A: frozen A arithmetic, direct compact NCCL packet storage.

The input is a verified nine-arm build's ``selective/a`` source tree. Nothing
in that build is edited. Only addressing/lifecycle changes; the original A
encoder, decoder arithmetic and CTA order remain byte-for-byte otherwise.
"""
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTOCOL = 'rtx5090-op1-a-direct-packet-v1'
PACKET_LAYOUT = 'compact-peer-soa-h128-v1'
PACKET_BYTES = 256 * 32 * 136
NATIVE_API = 'rtx5090_op1_a_direct_v1'
SCRATCH_POLICY = 'op1-a-fixed-stream-shared-tx-local-private-rx-v1'


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('OP1 A source contract changed: ' + old[:160])
    return source.replace(old, new, 1)


def peer_slot(rank, peer):
    """Rank-ordered packet slot, with the local BF16 contribution omitted."""
    if type(rank) is not int or type(peer) is not int or not (0 <= rank < 8 and 0 <= peer < 8):
        raise ValueError('Expected TP8 rank and peer')
    if peer == rank:
        raise ValueError('Local contribution has no quantized packet')
    return peer - int(peer > rank)


def wire_layout(rank):
    if type(rank) is not int or not 0 <= rank < 8:
        raise ValueError('Expected TP8 rank')
    splits = [0 if peer == rank else PACKET_BYTES for peer in range(8)]
    return dict(tx_splits=list(splits), rx_splits=list(splits),
                tx_bytes=7 * PACKET_BYTES, rx_bytes=7 * PACKET_BYTES)


def identity():
    return dict(protocol=PROTOCOL, native_api=NATIVE_API, packet_layout=PACKET_LAYOUT,
                quantization='all_remote', local_contribution='bf16',
                decoder='original-A-ring-bf16-rounding', prefetch=False,
                consumer_fusion=False, configuration_scope='instance',
                output_ownership='independent', gemm_submission_cache=False,
                packet_pack_kernels=0, packet_unpack_kernels=0,
                source_sha256={name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                               for name in ('op1_a_native.py', 'nccl_transport.py', 'local_transport.cu')})


def native_source(source):
    """Remove tile repacking; retain the checked local C ABI and CUDA lifetime."""
    if '#define RTX5090_LOCAL_CONSUMER 0\n#define RTX5090_LOCAL_SELECTIVE 0\n' not in source:
        raise ValueError('OP1 A requires the baseline A nonselective source')
    begin = source.index('// Packet bytes are copied without any FP arithmetic.')
    end = source.index('struct Local {', begin)
    source = source[:begin] + '// Producer and decoder use NCCL wire storage directly.\n' + source[end:]
    source = replace_once(source,
        'producer.peers[destination] = static_cast<unsigned char*>(tx_packets) + destination * kPacket;',
        'const int compact = destination == rank ? 0 : destination - int(destination > rank);\n'
        '      producer.peers[destination] = static_cast<unsigned char*>(tx_wire) + compact * kPacket;')
    source = replace_once(source,
        '// Frozen decoders index peers[rank] + source * packet_bytes.',
        '// The A decoder only changes source addressing to compact rank order.')
    source = replace_once(source,
        'receiver.peers[peer] = static_cast<unsigned char*>(rx_packets);',
        'receiver.peers[peer] = static_cast<unsigned char*>(rx_wire);')
    begin = source.index('void validate_layout(Local& local, int64_t tx_bytes, int64_t rx_bytes) {')
    end = source.index('\nstruct RestoreTaco', begin)
    source = source[:begin] + '''void validate_layout(Local& local, int64_t tx_bytes, int64_t rx_bytes) {
  check(!local.selected && !local.tx_offsets && !local.rx_offsets,
        "OP1 A has no selective mask or tile offset buffers");
  check(tx_bytes == (kWorld - 1) * kPacket && rx_bytes == tx_bytes,
        "OP1 A requires seven complete rank-ordered SoA packets");
  check(local.tx_packets == local.tx_wire && local.rx_packets == local.rx_wire,
        "OP1 A packet storage must alias the corresponding NCCL wire storage");
  check(local.tx_bf16 == local.rx_bf16,
        "OP1 A keeps only one full BF16 local-contribution workspace");
}
''' + source[end:]
    source = replace_once(source, 'rtx5090_local_protocol_version() { return 1; }',
                          'rtx5090_local_protocol_version() { return 2; }')
    source = replace_once(source,
        'local_pointer(tx_offsets, local->device); local_pointer(rx_offsets, local->device);',
        'check(!tx_offsets && !rx_offsets, "OP1 A has no tile offset buffers");')
    source = replace_once(source,
        'check(tx_bf16 != rx_bf16 && tx_packets != rx_packets && tx_wire != rx_wire,',
        'check(tx_packets != rx_packets && tx_wire != rx_wire && rx_bf16 != tx_wire && rx_bf16 != rx_wire,')
    source = replace_once(source,
        'check(local->phase == 0 && !local->cache, "Reconfigure requires an idle uncached producer");\n'
        '    local->configure();\n    ++local->reconfigurations;',
        'check(false, "OP1 A configuration is immutable after construction");')
    source = replace_once(source,
        '    rtx5090_pack_tiles_kernel<<<kTiles, 256, 0, local->stream>>>(local->rank, local->selected,\n'
        '        static_cast<const __nv_bfloat16*>(local->tx_bf16), static_cast<const unsigned char*>(local->tx_packets),\n'
        '        local->tx_offsets, static_cast<unsigned char*>(local->tx_wire));\n', '')
    source = replace_once(source,
        '    rtx5090_unpack_tiles_kernel<<<kTiles, 256, 0, local->stream>>>(local->rank, local->selected,\n'
        '        static_cast<__nv_bfloat16*>(local->rx_bf16), static_cast<unsigned char*>(local->rx_packets),\n'
        '        local->rx_offsets, static_cast<const unsigned char*>(local->rx_wire));\n'
        '    cuda_check(cudaGetLastError());\n', '')
    source = replace_once(source,
        'const int64_t state[] = {1, local->initializations, local->produces, local->decodes,',
        'const int64_t state[] = {2, local->initializations, local->produces, local->decodes,')
    return source


def frontend_source():
    """Standalone installed frontend; imports no controller or repository code."""
    source = (HERE / 'nccl_transport.py').read_text()
    source = replace_once(source, "PROTOCOL = 'rtx5090-nccl-staged-v2'", 'PROTOCOL = ' + repr(PROTOCOL))
    source = replace_once(source, "NATIVE_ARM = '__RTX5090_NATIVE_ARM__'", "NATIVE_ARM = 'a'")
    source = replace_once(source, "NATIVE_ARMS = ('a', 'r', 'c', 'cr3', 'ma89', 'ma11')", "NATIVE_ARMS = ('a',)")
    source = replace_once(source,
        "SCRATCH_POLICY = 'fixed-stream-shared-scratch-private-packets-v1'",
        'SCRATCH_POLICY = ' + repr(SCRATCH_POLICY))
    source = replace_once(source,
        "SHARED_BUFFERS = ('tx_bf16', 'tx_packets', 'rx_bf16', 'tx_wire', 'rx_wire')",
        "SHARED_BUFFERS = ('tx_wire', 'rx_bf16')")
    begin = source.index('def wire_layout(selected, rank):')
    end = source.index('\n\nclass _NativeRuntime:', begin)
    source = source[:begin] + '''def wire_layout(selected, rank):
    require(selected is None, 'OP1 A only supports all-remote quantization')
    require(type(rank) is int and 0 <= rank < 8, 'Expected a TP8 rank')
    splits = [0 if peer == rank else PACKET_BYTES_PER_SOURCE for peer in range(8)]
    return dict(tx_splits=list(splits), rx_splits=list(splits),
                tx_bytes=7*PACKET_BYTES_PER_SOURCE, rx_bytes=7*PACKET_BYTES_PER_SOURCE)
''' + source[end:]
    source = replace_once(source, "self.lib.rtx5090_local_protocol_version() == 1", "self.lib.rtx5090_local_protocol_version() == 2")
    source = replace_once(source, "pointers += [buffers[key].data_ptr() for key in",
                          "pointers += [buffers[key].data_ptr() if key in buffers else None for key in")
    source = source.replace("native_api='rtx5090_local_v1'", "native_api=" + repr(NATIVE_API))
    source = replace_once(source, "with self.scope('produce_pack'):", "with self.scope('produce_direct'):")
    source = replace_once(source, "with self.scope('unpack_decode_consumer' if bias is not None else 'unpack_decode'):",
                          "with self.scope('decode_direct'):")
    source = replace_once(source, "producer_calls=native['producer_calls'], pack_calls=native['producer_calls'],",
                          "producer_calls=native['producer_calls'], pack_calls=0,")
    source = replace_once(source, "unpack_calls=native['decode_calls'], decode_calls=native['decode_calls'],",
                          "unpack_calls=0, decode_calls=native['decode_calls'],")
    source = replace_once(source, "configuration_scope='per_call' if self.reconfigure else 'instance',",
                          "configuration_scope='instance', packet_layout='compact-peer-soa-h128-v1',\n"
                          "                    output_ownership='independent', gemm_submission_cache=False,")
    source = replace_once(source, "self.owns_output = self.owns_configuration = NATIVE_ARM != 'a'",
                          "self.owns_output = self.owns_configuration = True")
    source = replace_once(source,
        "require((self.mask is None) == (NATIVE_ARM in ('a', 'r')), 'Native arm/selection family mismatch')",
        "require(self.mask is None, 'OP1 A only supports all-remote quantization')")
    begin = source.index('        def allocate_scratch():')
    end = source.index('        self.packet_bytes_per_source =', begin)
    source = source[:begin] + '''        def allocate_scratch():
            return dict(tx_wire=torch.empty((layout['tx_bytes'],), dtype=torch.uint8, device=self.device),
                        rx_bf16=torch.empty((8, 256, 4096), dtype=torch.bfloat16, device=self.device))
        self._scratch = _SCRATCH_POOL.acquire(scratch_key, allocate_scratch, group)
        self.buffers = dict(self._scratch.buffers)
        self.buffers['rx_wire'] = torch.empty((layout['rx_bytes'],), dtype=torch.uint8, device=self.device)
        self.buffers['tx_packets'] = self.buffers['tx_wire']
        self.buffers['rx_packets'] = self.buffers['rx_wire']
        self.buffers['tx_bf16'] = self.buffers['rx_bf16']
''' + source[end:]
    source = replace_once(source,
        "native = _NativeRuntime(library, torch, self.rank, k_local, NATIVE_ARM in ('cr3', 'ma89', 'ma11'),",
        'native = _NativeRuntime(library, torch, self.rank, k_local, False,')
    source = replace_once(source, "reconfigure=NATIVE_ARM == 'a',", 'reconfigure=False,')
    # Remove unused double-buffering/build helpers from the deployed module.
    source = source[:source.index('\n\nclass GemmRSTacoDoubleBuffered:')]
    source = source.replace('unpacking/decoding', 'direct decoding').replace(
        'Receive packets, selection masks, native handles/GEMM caches and returned',
        'Receive packets, native handles and returned')
    return source


def overrides(source_root):
    """Return isolated overrides for a verified baseline ``selective/a`` tree."""
    root = Path(source_root)
    runtime = (root / 'src/gemm_rs/taco_runtime.cu').read_text()
    runtime = replace_once(runtime,
        'const unsigned char* slot = c.peers[c.rank] + src * source_bytes;',
        'const unsigned char* slot = c.peers[c.rank] + (src - int(src > c.rank)) * source_bytes;')
    # The separate encoder is unused, but do not leave an exported API pointing
    # at the historical peer layout in the isolated direct-packet library.
    runtime = replace_once(runtime,
        'unsigned char* slot = c.peers[dst] + c.rank*taco_source_bytes(c);',
        'unsigned char* slot = c.peers[dst];  // Local destination packet, already compacted.')
    epilogue = (root / 'src/gemm_rs/epilogue_evt.hpp').read_text()
    if 'unsigned char* slot = c.peers[dst_rank];' not in epilogue:
        raise ValueError('OP1 A requires the local producer epilogue; no peer pointer migration is allowed')
    return {'src/gemm_rs/taco_runtime.cu': runtime,
            'src/gemm_rs/local_transport.cu': native_source((root / 'src/gemm_rs/local_transport.cu').read_text()),
            'python/flux/gemm_rs_taco.py': frontend_source()}
