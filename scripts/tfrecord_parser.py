"""Pure-protobuf TFRecord parser. No tensorflow dependency.

Parses tf.train.Example records from TFRecord files using only
google.protobuf internals. Works with numpy 2.x where tensorflow fails.
"""
import struct
from google.protobuf.internal.decoder import _DecodeVarint32


def read_tfrecords(path):
    """Yield raw record bytes from a TFRecord file."""
    with open(path, 'rb') as f:
        while True:
            header = f.read(8)
            if len(header) < 8:
                break
            length = struct.unpack('Q', header)[0]
            f.read(4)  # length CRC
            data = f.read(length)
            f.read(4)  # data CRC
            yield data


def parse_tf_example(data):
    """Parse a tf.train.Example from raw protobuf bytes.

    Returns dict mapping feature name -> dict with one key:
        'bytes_list': list of bytes
        'float_list': list of float
        'int64_list': list of int
    """
    pos = 0
    while pos < len(data):
        tag, new_pos = _DecodeVarint32(data, pos)
        pos = new_pos
        fn = tag >> 3
        wt = tag & 0x7
        if wt == 2 and fn == 1:  # Features message
            length, new_pos = _DecodeVarint32(data, pos)
            pos = new_pos
            return _parse_features(data[pos:pos + length])
        else:
            pos = _skip_field(data, pos, wt)
    return {}


def _parse_features(data):
    features = {}
    pos = 0
    while pos < len(data):
        tag, new_pos = _DecodeVarint32(data, pos)
        pos = new_pos
        fn = tag >> 3
        wt = tag & 0x7
        if wt == 2 and fn == 1:  # map entry
            length, new_pos = _DecodeVarint32(data, pos)
            pos = new_pos
            key, value = _parse_map_entry(data[pos:pos + length])
            if key is not None:
                features[key] = value
            pos += length
        else:
            pos = _skip_field(data, pos, wt)
    return features


def _parse_map_entry(data):
    key = None
    value = None
    pos = 0
    while pos < len(data):
        tag, new_pos = _DecodeVarint32(data, pos)
        pos = new_pos
        fn = tag >> 3
        wt = tag & 0x7
        if wt == 2:
            length, new_pos = _DecodeVarint32(data, pos)
            pos = new_pos
            if fn == 1:
                key = data[pos:pos + length].decode('utf-8')
            elif fn == 2:
                value = _parse_feature(data[pos:pos + length])
            pos += length
        else:
            pos = _skip_field(data, pos, wt)
    return key, value


def _parse_feature(data):
    pos = 0
    while pos < len(data):
        tag, new_pos = _DecodeVarint32(data, pos)
        pos = new_pos
        fn = tag >> 3
        wt = tag & 0x7
        if wt == 2:
            length, new_pos = _DecodeVarint32(data, pos)
            pos = new_pos
            field_data = data[pos:pos + length]
            pos += length
            if fn == 1:
                return {'bytes_list': _parse_bytes_list(field_data)}
            elif fn == 2:
                return {'float_list': _parse_float_list(field_data)}
            elif fn == 3:
                return {'int64_list': _parse_int64_list(field_data)}
        else:
            pos = _skip_field(data, pos, wt)
    return {}


def _parse_bytes_list(data):
    values = []
    pos = 0
    while pos < len(data):
        tag, new_pos = _DecodeVarint32(data, pos)
        pos = new_pos
        fn = tag >> 3
        wt = tag & 0x7
        if wt == 2 and fn == 1:
            length, new_pos = _DecodeVarint32(data, pos)
            pos = new_pos
            values.append(data[pos:pos + length])
            pos += length
        else:
            pos = _skip_field(data, pos, wt)
    return values


def _parse_float_list(data):
    pos = 0
    while pos < len(data):
        tag, new_pos = _DecodeVarint32(data, pos)
        pos = new_pos
        fn = tag >> 3
        wt = tag & 0x7
        if wt == 2 and fn == 1:  # packed floats
            length, new_pos = _DecodeVarint32(data, pos)
            pos = new_pos
            n = length // 4
            values = list(struct.unpack(f'<{n}f', data[pos:pos + length]))
            pos += length
            return values
        else:
            pos = _skip_field(data, pos, wt)
    return []


def _parse_int64_list(data):
    pos = 0
    while pos < len(data):
        tag, new_pos = _DecodeVarint32(data, pos)
        pos = new_pos
        fn = tag >> 3
        wt = tag & 0x7
        if wt == 2 and fn == 1:  # packed varints
            length, new_pos = _DecodeVarint32(data, pos)
            pos = new_pos
            values = []
            end = pos + length
            while pos < end:
                v, pos = _DecodeVarint32(data, pos)
                values.append(v)
            return values
        else:
            pos = _skip_field(data, pos, wt)
    return []


def _skip_field(data, pos, wire_type):
    if wire_type == 0:  # varint
        _, pos = _DecodeVarint32(data, pos)
    elif wire_type == 1:  # 64-bit
        pos += 8
    elif wire_type == 2:  # length-delimited
        length, new_pos = _DecodeVarint32(data, pos)
        pos = new_pos + length
    elif wire_type == 5:  # 32-bit
        pos += 4
    return pos
