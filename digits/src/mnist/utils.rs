use std::io::Read;

pub fn read_u8<T: Read>(source: &mut T) -> u8 {
    let mut buf: Vec<u8> = Vec::new();
    buf.resize(1, 0);

    let bytes_read = source.read(&mut buf).expect("failed to read into buffer");

    assert!(bytes_read == 1, "failed to read another 1 byte from source");

    buf[0]
}

pub fn read_u32<T: Read>(source: &mut T) -> u32 {
    let mut buf: Vec<u8> = Vec::new();
    buf.resize(4, 0);

    let bytes_read = source.read(&mut buf).expect("failed to read into buffer");

    assert!(
        bytes_read == 4,
        "failed to read another 4 bytes from source"
    );

    convert_byte_array_to_number(&buf)
}

pub fn read_vec_u8<T: Read>(source: &mut T, size: u32) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    buf.resize(size as usize, 0);
    source
        .read_exact(&mut buf)
        .expect("failed to read into buffer");

    buf
}

pub fn convert_byte_array_to_number(buf: &Vec<u8>) -> u32 {
    if buf.len() != 4 {
        panic!("invalid vector size. should be 4 bytes");
    }

    ((buf[0] as u32) << 24) + ((buf[1] as u32) << 16) + ((buf[2] as u32) << 8) + (buf[3] as u32)
}
