use std::f64::EPSILON;

use ncollide2d::na::{clamp, Rotation2, Vector2};

use crate::primitives::{circle::Circle, pos::VirtualPosition, rect::Rect};

use super::utils::{
    map_circle_to_collision_shape, map_collision_pos_to_pos, map_rect_to_collision_shape,
};

pub struct CircleRectCollision {
    // position on rectangle shape where circle touches rectangle
    pub rect_point: VirtualPosition,

    // normal angle for rectangle in contact position
    pub rect_norm: f64,

    // circle position where it should be right before touch.
    // starting from this position circle may bounce out of rectangle
    pub circle_center: VirtualPosition,
}

pub fn collide_circle_and_rect(circle: &Circle, rect: &Rect) -> Option<CircleRectCollision> {
    let (ball, ball_pos) = map_circle_to_collision_shape(circle);
    let (cuboid, cuboid_pos) = map_rect_to_collision_shape(rect);

    let contact = ncollide2d::query::contact(&cuboid_pos, &cuboid, &ball_pos, &ball, 0.0);

    contact.map(|c| {
        let rect_point = map_collision_pos_to_pos(c.world1.x, c.world1.y);
        let rect_norm = Rotation2::rotation_between(&Vector2::x(), &c.normal).angle();
        let circle_center =
            get_circle_position_to_almost_touch(&rect_point, rect_norm, circle.radius());

        CircleRectCollision {
            rect_point,
            rect_norm,
            circle_center,
        }
    })
}

fn get_circle_position_to_almost_touch(
    point: &VirtualPosition,
    norm: f64,
    circle_radius: f64,
) -> VirtualPosition {
    let circle_center_x = point.x() + norm.cos() * (circle_radius + EPSILON);
    let circle_center_y = point.y() - norm.sin() * (circle_radius + EPSILON);

    // clamp to valid position in case circle goes deep out of bounds
    let circle_center_x = clamp(circle_center_x, circle_radius, 1.0 - circle_radius);
    let circle_center_y = clamp(circle_center_y, circle_radius, 1.0 - circle_radius);

    VirtualPosition::new(circle_center_x, circle_center_y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::dimensions::Dimensions;
    use approx::assert_abs_diff_eq;
    use std::f64::{consts::PI, EPSILON};

    #[test]
    pub fn test_collide_circle_and_rect_to_the_right() {
        let rect = Rect::new(
            VirtualPosition::new(0.5 + EPSILON, 0.0),
            Dimensions::new(0.5 - EPSILON, 1.0),
        );

        // no touch
        let res = collide_circle_and_rect(&Circle::new(VirtualPosition::new(0.3, 0.5), 0.2), &rect);

        assert!(res.is_none());

        // touch
        let res =
            collide_circle_and_rect(&Circle::new(VirtualPosition::new(0.31, 0.5), 0.2), &rect);

        assert_abs_diff_eq!(res.as_ref().unwrap().rect_point.x(), 0.5);
        assert_abs_diff_eq!(res.as_ref().unwrap().rect_point.y(), 0.5);
        assert_abs_diff_eq!(res.as_ref().unwrap().rect_norm, PI);
        assert_abs_diff_eq!(res.as_ref().unwrap().circle_center.x(), 0.3);
        assert_abs_diff_eq!(res.as_ref().unwrap().circle_center.y(), 0.5);

        // no touch
        let res = collide_circle_and_rect(&Circle::new(res.unwrap().circle_center, 0.2), &rect);

        assert!(res.is_none());
    }

    #[test]
    pub fn test_collide_circle_and_rect_above() {
        let rect = Rect::new(
            VirtualPosition::new(0.0, 0.0),
            Dimensions::new(1.0, 0.5 - EPSILON),
        );

        // no touch
        let res = collide_circle_and_rect(&Circle::new(VirtualPosition::new(0.5, 0.7), 0.2), &rect);

        assert!(res.is_none());

        // touch
        let res =
            collide_circle_and_rect(&Circle::new(VirtualPosition::new(0.5, 0.69), 0.2), &rect);

        assert_abs_diff_eq!(res.as_ref().unwrap().rect_point.x(), 0.5);
        assert_abs_diff_eq!(res.as_ref().unwrap().rect_point.y(), 0.5);
        assert_abs_diff_eq!(res.as_ref().unwrap().rect_norm, -PI / 2.0);
        assert_abs_diff_eq!(res.as_ref().unwrap().circle_center.x(), 0.5);
        assert_abs_diff_eq!(res.as_ref().unwrap().circle_center.y(), 0.7);

        // no touch
        let res = collide_circle_and_rect(&Circle::new(res.unwrap().circle_center, 0.2), &rect);

        assert!(res.is_none());
    }

    #[test]
    pub fn test_collide_circle_and_rect_to_the_left() {
        let rect = Rect::new(
            VirtualPosition::new(0.0, 0.0),
            Dimensions::new(0.5 - EPSILON, 1.0),
        );

        // no touch
        let res = collide_circle_and_rect(&Circle::new(VirtualPosition::new(0.7, 0.5), 0.2), &rect);

        assert!(res.is_none());

        // touch
        let res =
            collide_circle_and_rect(&Circle::new(VirtualPosition::new(0.69, 0.5), 0.2), &rect);

        assert_abs_diff_eq!(res.as_ref().unwrap().rect_point.x(), 0.5);
        assert_abs_diff_eq!(res.as_ref().unwrap().rect_point.y(), 0.5);
        assert_abs_diff_eq!(res.as_ref().unwrap().rect_norm, 0.0);
        assert_abs_diff_eq!(res.as_ref().unwrap().circle_center.x(), 0.7);
        assert_abs_diff_eq!(res.as_ref().unwrap().circle_center.y(), 0.5);

        // no touch
        let res = collide_circle_and_rect(&Circle::new(res.unwrap().circle_center, 0.2), &rect);

        assert!(res.is_none());
    }

    #[test]
    pub fn test_collide_circle_and_rect_below() {
        let rect = Rect::new(
            VirtualPosition::new(0.0, 0.5 + EPSILON),
            Dimensions::new(1.0, 0.5 - EPSILON),
        );

        // no touch
        let res = collide_circle_and_rect(&Circle::new(VirtualPosition::new(0.5, 0.3), 0.2), &rect);

        assert!(res.is_none());

        // touch
        let res =
            collide_circle_and_rect(&Circle::new(VirtualPosition::new(0.5, 0.31), 0.2), &rect);

        assert_abs_diff_eq!(res.as_ref().unwrap().rect_point.x(), 0.5);
        assert_abs_diff_eq!(res.as_ref().unwrap().rect_point.y(), 0.5);
        assert_abs_diff_eq!(res.as_ref().unwrap().rect_norm, PI / 2.0);
        assert_abs_diff_eq!(res.as_ref().unwrap().circle_center.x(), 0.5);
        assert_abs_diff_eq!(res.as_ref().unwrap().circle_center.y(), 0.3);

        // no touch
        let res = collide_circle_and_rect(&Circle::new(res.unwrap().circle_center, 0.2), &rect);

        assert!(res.is_none());
    }
}
