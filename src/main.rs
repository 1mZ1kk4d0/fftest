use std::{sync::{LazyLock, Mutex, MutexGuard}, thread, time::Duration};

use opencv::{
    core::{get_build_information, get_cuda_enabled_device_count, print_cuda_device_info, Mat, MatTrait, MatTraitConst, Point, Rect, Scalar, UMat, UMatTrait, UMatUsageFlags}, dnn, highgui, imgcodecs::{self, imread}, imgproc::{self, FONT_HERSHEY_COMPLEX, LINE_8}
};
use scap::{
    capturer::{Capturer, Options, Resolution},
    frame::{Frame, FrameType},
    get_all_targets, Target,
};
use testes::{detect_image::{self, detect_tree, detect_tree_v2}, get_mutable_model, init_dnn};

mod testes;



#[tokio::main]
async fn main() -> opencv::Result<()> {

    println!("{}", get_build_information()?);


    let targets = get_all_targets();

    let screen = targets.iter().find(|t| 

        match t {
          Target::Window(jan) => jan.title == "Title Window",
          _ => false
        }
    ).cloned();


    if let None = screen {
        println!("Window not found.");
    }


    let options = Options {
        fps: 15,
        target: screen,
        show_cursor: true,
        show_highlight: false,
        excluded_targets: None,
        output_type: FrameType::BGRAFrame,
        output_resolution: Resolution::_480p,
        crop_area: None,
        ..Default::default()
    };

    let mut capturer = Capturer::new(options);

    capturer.start_capture();


    init_dnn();

    while let Ok(frame) = capturer.get_next_frame() {

        if let Frame::BGRA(bgra_frame) = frame {

            let data = bgra_frame.data;
            let rows = bgra_frame.height as i32;

            let bgra_mat = Mat::from_slice(&data)?;
            let bgra_mat = bgra_mat.reshape(4, rows)?;

            let mut bgr_mat = Mat::default();

            imgproc::cvt_color(&bgra_mat, &mut bgr_mat, imgproc::COLOR_BGRA2BGR, 0)?;


          detect_tree_v2(&mut bgr_mat)?;

            highgui::imshow("WindowNameLOL", &bgr_mat)?;

            if highgui::poll_key()? == 'q' as i32 {
                capturer.stop_capture();
                break;
            }
        }
    }

    Ok(())
}
