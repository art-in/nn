use std::path::PathBuf;

use plotters::prelude::*;

pub fn plot_losses(losses: &Vec<f64>, errors_percents: &[f64], dir: &str) {
    let path = PathBuf::from(dir).join("losses.bmp").as_path().to_owned();

    let root = BitMapBackend::new(&path, (1280, 480)).into_drawing_area();
    root.fill(&WHITE).expect("failed to fill drawing area");

    let max_loss = losses.iter().copied().reduce(f64::max).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption("loss", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..losses.len(), 0f64..max_loss)
        .expect("failed to build cartesian 2D");

    chart.configure_mesh().draw().expect("failed to draw mesh");

    chart
        .draw_series(LineSeries::new(
            losses.iter().enumerate().map(|(idx, d)| (idx, *d)),
            RED,
        ))
        .expect("failed to draw series");

    chart
        .draw_series(LineSeries::new(
            errors_percents
                .iter()
                .enumerate()
                .map(|(idx, p)| (idx, *p * max_loss)),
            BLUE,
        ))
        .expect("failed to draw series");

    root.present().expect("failed to present");
}
