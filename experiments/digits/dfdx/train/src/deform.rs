use data_utils::data_aug::DeformationConfig;

// config with image deformation params for MNIST dataset augmentation
pub const DATASET_DEFORM_CFG: DeformationConfig = DeformationConfig {
    max_mls_grid_density: 3,
    max_mls_shift: 2.0,
    // rotate more to the left, to compensate prevailing right slope
    max_rotate_degree: (-35.0, 5.0),
    max_blur: 0.3,
    max_contrast: 200.0,
    max_scale: 0.25,
    max_shift: 5,
};
