// based on "mnist" v0.5.0 (https://crates.io/crates/mnist)
// copied/cropped because it panics when dataset size doesn't match hardcoded values (ie. 60.000/10.000),
// and I need to read handmade dataset with same format but different size

use byteorder::{BigEndian, ReadBytesExt};
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

static BASE_PATH: &str = "data/";
static TRN_IMG_FILENAME: &str = "train-images-idx3-ubyte";
static TRN_LBL_FILENAME: &str = "train-labels-idx1-ubyte";
static TST_IMG_FILENAME: &str = "t10k-images-idx3-ubyte";
static TST_LBL_FILENAME: &str = "t10k-labels-idx1-ubyte";
static IMG_MAGIC_NUMBER: u32 = 0x0000_0803;
static LBL_MAGIC_NUMBER: u32 = 0x0000_0801;
static CLASSES: usize = 10;
static ROWS: usize = 28;
static COLS: usize = 28;

#[derive(Debug)]
pub struct Mnist {
    pub trn_img: Vec<u8>,
    pub trn_lbl: Vec<u8>,
    pub tst_img: Vec<u8>,
    pub tst_lbl: Vec<u8>,
}

#[derive(Debug)]
pub struct MnistBuilder<'a> {
    lbl_format: LabelFormat,
    base_path: &'a str,
    trn_img_filename: &'a str,
    trn_lbl_filename: &'a str,
    tst_img_filename: &'a str,
    tst_lbl_filename: &'a str,
}

impl<'a> MnistBuilder<'a> {
    pub fn new() -> MnistBuilder<'a> {
        MnistBuilder {
            lbl_format: LabelFormat::Digit,
            base_path: BASE_PATH,
            trn_img_filename: TRN_IMG_FILENAME,
            trn_lbl_filename: TRN_LBL_FILENAME,
            tst_img_filename: TST_IMG_FILENAME,
            tst_lbl_filename: TST_LBL_FILENAME,
        }
    }

    pub fn base_path(&mut self, base_path: &'a str) -> &mut MnistBuilder<'a> {
        self.base_path = base_path;
        self
    }

    pub fn finalize(&self) -> Mnist {
        let (trn_img, trn_len) = images(&Path::new(self.base_path).join(self.trn_img_filename));
        let (mut trn_lbl, trn_lbl_len) =
            labels(&Path::new(self.base_path).join(self.trn_lbl_filename));
        let (tst_img, tst_len) = images(&Path::new(self.base_path).join(self.tst_img_filename));
        let (mut tst_lbl, tst_lbl_len) =
            labels(&Path::new(self.base_path).join(self.tst_lbl_filename));

        assert_eq!(trn_len, trn_lbl_len);
        assert_eq!(tst_len, tst_lbl_len);

        if self.lbl_format == LabelFormat::OneHotVector {
            fn digit2one_hot(v: Vec<u8>) -> Vec<u8> {
                v.iter()
                    .flat_map(|&i| {
                        let mut v = vec![0; CLASSES as usize];
                        v[i as usize] = 1;
                        v
                    })
                    .collect()
            }
            trn_lbl = digit2one_hot(trn_lbl);
            tst_lbl = digit2one_hot(tst_lbl);
        }

        Mnist {
            trn_img,
            trn_lbl,
            tst_img,
            tst_lbl,
        }
    }
}

impl Default for MnistBuilder<'_> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, PartialEq)]
enum LabelFormat {
    Digit,
    OneHotVector,
}

fn labels(path: &Path) -> (Vec<u8>, usize) {
    let mut file =
        File::open(path).unwrap_or_else(|_| panic!("Unable to find path to labels at {:?}.", path));
    let magic_number = file
        .read_u32::<BigEndian>()
        .unwrap_or_else(|_| panic!("Unable to read magic number from {:?}.", path));
    assert!(
        LBL_MAGIC_NUMBER == magic_number,
        "Expected magic number {LBL_MAGIC_NUMBER} got {magic_number}.",
    );
    let length = file
        .read_u32::<BigEndian>()
        .unwrap_or_else(|_| panic!("Unable to length from {:?}.", path));
    (file.bytes().map(|b| b.unwrap()).collect(), length as usize)
}

fn images(path: &Path) -> (Vec<u8>, usize) {
    // Read whole file in memory
    let mut content: Vec<u8> = Vec::new();
    let mut file = {
        let mut fh = File::open(path)
            .unwrap_or_else(|_| panic!("Unable to find path to images at {:?}.", path));
        let _ = fh
            .read_to_end(&mut content)
            .unwrap_or_else(|_| panic!("Unable to read whole file in memory ({})", path.display()));
        // The read_u32() method, coming from the byteorder crate's ReadBytesExt trait, cannot be
        // used with a `Vec` directly, it requires a slice.
        &content[..]
    };

    let magic_number = file
        .read_u32::<BigEndian>()
        .unwrap_or_else(|_| panic!("Unable to read magic number from {:?}.", path));
    assert!(
        IMG_MAGIC_NUMBER == magic_number,
        "Expected magic number {IMG_MAGIC_NUMBER} got {magic_number}.",
    );
    let length = file
        .read_u32::<BigEndian>()
        .unwrap_or_else(|_| panic!("Unable to length from {:?}.", path));
    let rows = file
        .read_u32::<BigEndian>()
        .unwrap_or_else(|_| panic!("Unable to number of rows from {:?}.", path))
        as usize;
    assert!(ROWS == rows, "Expected rows length of {ROWS} got {rows}.",);
    let cols = file
        .read_u32::<BigEndian>()
        .unwrap_or_else(|_| panic!("Unable to number of columns from {:?}.", path))
        as usize;
    assert!(COLS == cols, "Expected cols length of {COLS} got {cols}.",);
    // Convert `file` from a Vec to a slice.

    (file.to_vec(), length as usize)
}
