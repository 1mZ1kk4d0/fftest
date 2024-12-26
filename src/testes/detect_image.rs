use opencv::{
    core::{Mat, MatTrait, MatTraitConst, MatTraitConstManual, Point, Rect, Scalar, Size, Vector, CV_32F},
    dnn::{self, Net, NetTrait, NetTraitConst},
    imgcodecs::{self, imread},
    imgproc::{self, TemplateMatchModes, TM_CCOEFF_NORMED, TM_SQDIFF},
};

use super::get_mutable_model;

pub fn detect_tree(bgr_mat: &mut Mat, template: Mat) -> opencv::Result<()> {
    let mut result = Mat::default();

    imgproc::match_template(
        bgr_mat,
        &template,
        &mut result,
        4,
        &opencv::core::no_array(),
    )?;

    // Encontra o local de máxima correspondência
    let mut min_val = 0.0;
    let mut max_val = 0.0;
    let mut min_loc = Point::default();
    let mut max_loc = Point::default();

    opencv::core::min_max_loc(
        &result,
        Some(&mut min_val),
        Some(&mut max_val),
        Some(&mut min_loc),
        Some(&mut max_loc),
        &opencv::core::no_array(),
    )?;

    let threshold = 0.8;


    let top_left = max_loc;

    imgproc::rectangle(
        bgr_mat,
        Rect::new(top_left.x, top_left.y, template.cols(), template.rows()),
        Scalar::new(0.0, 0.0, 255.0, 0.0),
        10,
        imgproc::LINE_8,
        0,
    )?;

    Ok(())
}


pub fn detect_tree_v2(mat: &mut Mat) -> opencv::Result<()> {

    let mut net = get_mutable_model();


    let input_blob = dnn::blob_from_image(
        mat,      
        1.0,
        Size::new(640, 640),
        opencv::core::Scalar::all(0.0),   
        false,
        false,
        opencv::core::CV_32F
    )?;



    net.set_input(&input_blob, "images", 1.0, Scalar::default())?;

    let out_layers = net.get_unconnected_out_layers_names()?;

    let mut detections = Mat::default();
 
    net.forward(&mut detections, &out_layers).unwrap();



    

    Ok(())
}