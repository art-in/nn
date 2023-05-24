use ncollide2d::{
    na::{self, Isometry2, Vector2},
    shape::{Ball, Cuboid},
};

use crate::primitives::{circle::Circle, pos::VirtualPosition, rect::Rect};

// map position to vector with origin (0;0) in bottom-left corner
pub fn map_pos_to_collision_vector(pos: VirtualPosition) -> Vector2<f64> {
    Vector2::new(pos.x(), 1.0 - pos.y())
}

pub fn map_collision_pos_to_pos(x: f64, y: f64) -> VirtualPosition {
    VirtualPosition::new_unchecked(x, 1.0 - y)
}

pub fn map_rect_to_collision_shape(rect: &Rect) -> (Cuboid<f64>, Isometry2<f64>) {
    let half_width = rect.dim().width() / 2.0;
    let half_height = rect.dim().height() / 2.0;

    let cuboid = Cuboid::new(Vector2::new(half_width, half_height));

    let cuboid_pos = Isometry2::new(
        map_pos_to_collision_vector(VirtualPosition::new_unchecked(
            rect.pos().x() + half_width,
            rect.pos().y() + half_height,
        )),
        na::zero(),
    );

    (cuboid, cuboid_pos)
}

pub fn map_circle_to_collision_shape(circle: &Circle) -> (Ball<f64>, Isometry2<f64>) {
    let ball = Ball::new(circle.radius());
    let ball_pos = Isometry2::new(
        map_pos_to_collision_vector(VirtualPosition::new_unchecked(
            circle.pos().x(),
            circle.pos().y(),
        )),
        na::zero(),
    );

    (ball, ball_pos)
}
