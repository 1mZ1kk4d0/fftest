use scap::{
    capturer::{Capturer, Options, Resolution},
    frame::{Frame, FrameType},
};

use opencv::{
    core::{
        get_build_information, get_cuda_enabled_device_count, GpuMat, GpuMatTrait,
        GpuMatTraitConst, Mat, MatTraitConst, Mat_AUTO_STEP, Rect, Scalar, Size, CV_8UC4,
    },
    highgui,
    imgproc::{self, cvt_color},
    videoio::{VideoWriter, VideoWriterTrait},
};

mod testes;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("OpenCV version: {:?}", get_build_information().unwrap());

    if get_cuda_enabled_device_count()? <= 0 {
        println!("Cuda não encontrado.");
    } else {
        println!("Cuda encontrado.");
    }

    let options = Options {
        fps: 30,
        target: None,
        show_cursor: false,
        show_highlight: true,
        excluded_targets: None,
        output_type: FrameType::BGRAFrame,
        output_resolution: Resolution::Captured,
        crop_area: None,
        ..Default::default()
    };

    let mut capturer = Capturer::new(options);
    capturer.start_capture();

    highgui::named_window("Scap Preview", highgui::WINDOW_NORMAL)?;

    loop {
        if let Ok(frame) = capturer.get_next_frame() {
            if let Frame::BGRA(bgra_frame) = frame {
                let data = bgra_frame.data.as_slice();
                let width = bgra_frame.width as i32;
                let height = bgra_frame.height as i32;

                let bgra_mat = Mat::from_slice(data)?;
                let bgra_mat = bgra_mat.reshape(4, height)?;

                let mut bgr_frame = Mat::default();
                imgproc::cvt_color(&bgra_mat, &mut bgr_frame, imgproc::COLOR_BGRA2BGR, 0)?;

                let top_left = (100, 50);
                let bottom_right = (550, 150);
                let rect = Rect::new(
                    top_left.0,
                    top_left.1,
                    bottom_right.0 - top_left.0,
                    bottom_right.1 - top_left.1,
                );

                let color = Scalar::new(100.0, 0.0, 0.0, 0.0);
                let thickness = 2;
                let line_type = imgproc::LINE_8;

                imgproc::rectangle(&mut bgr_frame, rect, color, thickness, line_type, 0)?;
                highgui::imshow("Scap Preview", &bgr_frame)?;



                if highgui::wait_key(1)? == 'q' as i32 {
                    break;
                }
            }
        }
    }

    Ok(())
}
