use std::io::Read;

pub fn read_u8<T: Read>(source: &mut T) -> u8 {
    let mut buf: [u8; 1] = [0; 1];

    source
        .read_exact(&mut buf)
        .expect("failed to read into buffer");

    buf[0]
}

pub fn read_u32<T: Read>(source: &mut T) -> u32 {
    let mut buf: [u8; 4] = [0; 4];

    source
        .read_exact(&mut buf)
        .expect("failed to read into buffer");

    convert_byte_array_to_u32(&buf)
}

pub fn read_vec_u8<T: Read>(source: &mut T, size: u32) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    buf.resize(size as usize, 0);
    source
        .read_exact(&mut buf)
        .expect("failed to read into buffer");

    buf
}

pub fn convert_byte_array_to_u32(buf: &[u8; 4]) -> u32 {
    ((buf[0] as u32) << 24) + ((buf[1] as u32) << 16) + ((buf[2] as u32) << 8) + (buf[3] as u32)
}
