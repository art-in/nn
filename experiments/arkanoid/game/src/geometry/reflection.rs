use ncollide2d::na::{Rotation2, Vector2};

// http://www.sunshine2k.de/articles/coding/vectorreflection/vectorreflection.html
pub fn reflect_angle_by_normal(a: f64, norm: f64) -> f64 {
    if a == norm {
        return a;
    }

    let a = Vector2::new(a.cos(), a.sin());
    let norm = Vector2::new(norm.cos(), norm.sin());

    let reflection = a - (2.0 * a.dot(&norm) * norm);

    Rotation2::rotation_between(&Vector2::x(), &reflection).angle()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::f64::consts::PI;

    #[test]
    pub fn test_reflect_angle() {
        // normal points to the left
        let left = PI;
        assert_abs_diff_eq!(reflect_angle_by_normal(0.0, left), PI);
        assert_abs_diff_eq!(reflect_angle_by_normal(PI, left), PI);
        assert_abs_diff_eq!(reflect_angle_by_normal(PI / 2.0, left), PI / 2.0);
        assert_abs_diff_eq!(reflect_angle_by_normal(PI / 4.0, left), PI * 0.75);
        assert_abs_diff_eq!(
            reflect_angle_by_normal(-PI / 4.0, left),
            -PI * 0.75,
            epsilon = 1e-9
        );

        // normal points to the right
        let right = 0.0;
        assert_abs_diff_eq!(reflect_angle_by_normal(PI, right), 0.0);
        assert_abs_diff_eq!(reflect_angle_by_normal(0.0, right), 0.0);
        assert_abs_diff_eq!(reflect_angle_by_normal(PI / 2.0, right), PI / 2.0);
        assert_abs_diff_eq!(reflect_angle_by_normal(PI * 0.75, right), PI * 0.25);
        assert_abs_diff_eq!(reflect_angle_by_normal(-PI * 0.75, right), -PI * 0.25);

        // normal points up
        let up = PI / 2.0;
        assert_abs_diff_eq!(reflect_angle_by_normal(-PI / 2.0, up), PI / 2.0);
        assert_abs_diff_eq!(reflect_angle_by_normal(PI / 2.0, up), PI / 2.0);
        assert_abs_diff_eq!(reflect_angle_by_normal(0.0, up), 0.0);
        assert_abs_diff_eq!(reflect_angle_by_normal(-PI * 0.75, up), PI * 0.75);
        assert_abs_diff_eq!(reflect_angle_by_normal(-PI * 0.25, up), PI * 0.25);

        // normal points down
        let down = -PI / 2.0;
        assert_abs_diff_eq!(reflect_angle_by_normal(PI / 2.0, down), -PI / 2.0);
        assert_abs_diff_eq!(reflect_angle_by_normal(-PI / 2.0, down), -PI / 2.0);
        assert_abs_diff_eq!(reflect_angle_by_normal(0.0, down), 0.0);
        assert_abs_diff_eq!(reflect_angle_by_normal(PI * 0.75, down), -PI * 0.75);
        assert_abs_diff_eq!(reflect_angle_by_normal(PI * 0.25, down), -PI * 0.25);
    }
}
