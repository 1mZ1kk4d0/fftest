use scap::{
    capturer::{Capturer, Options, Resolution},
    frame::{Frame, FrameType},
};

use opencv::{
    core::{
        get_build_information, get_cuda_enabled_device_count, GpuMat, GpuMatTrait, GpuMatTraitConst, Mat, MatTraitConst, Mat_AUTO_STEP, Rect, Scalar, Size, UMat, UMatTraitConst, UMatUsageFlags, ACCESS_RW, CV_8UC4
    }, highgui, imgcodecs, imgproc::{self, cvt_color}, videoio::{VideoWriter, VideoWriterTrait}
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
        fps: 260,
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

        let template = imgcodecs::imread("tree_template.png", imgcodecs::IMREAD_COLOR)?;
        let template_width = template.cols();
        let template_height = template.rows();

        if let Ok(frame) = capturer.get_next_frame() {
            if let Frame::BGRA(bgra_frame) = frame {
                let data = bgra_frame.data.as_slice();
                let width = bgra_frame.width as i32;
                let height = bgra_frame.height as i32;
        
                // Criar Mat com os dados da frame
                let bgra_mat = Mat::from_slice(data)?;
                let bgra_mat = bgra_mat.reshape(4, height)?;
        
                // Converter para UMat (memória da GPU)
                let mut bgra_umat = UMat::new(UMatUsageFlags::USAGE_DEFAULT);
                bgra_mat.copy_to(&mut bgra_umat)?;
        
                // Conversão de cor na GPU
                let mut bgr_umat = UMat::new(UMatUsageFlags::USAGE_DEFAULT);
                imgproc::cvt_color(&bgra_umat, &mut bgr_umat, imgproc::COLOR_BGRA2BGR, 0)?;
        
                // Transferir os dados de volta para um Mat na CPU para desenhar
                let mut bgr_frame = bgr_umat.get_mat(opencv::core::AccessFlag::ACCESS_RW)?;
        
                let top_left = (50, 50);
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

                //bgra.
        
                if highgui::poll_key()? == 'q' as i32 {
                    break;
                }
            }
        }
        
    }

    Ok(())
}
