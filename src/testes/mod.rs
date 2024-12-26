use opencv::{
    core::{
        get_build_information, get_cuda_enabled_device_count, print_cuda_device_info, Mat,
        MatTrait, MatTraitConst, Point, Rect, Scalar, UMat, UMatTrait, UMatUsageFlags,
    },
    dnn::{self, NetTrait, NetTraitConst},
    highgui,
    imgcodecs::{self, imread},
    imgproc::{self, FONT_HERSHEY_COMPLEX, LINE_8},
};
use std::sync::{LazyLock, Mutex, MutexGuard};

pub mod detect_image;

pub static MODEL: LazyLock<Mutex<dnn::Net>> = LazyLock::new(|| {
    println!("Loading ONNX model");

    let net = dnn::read_net_from_onnx("./best.onnx");

    match net {
        Ok(net) => Mutex::new(net),
        Err(e) => {
            eprintln!("Failed to load the ONNX model: {:?}", e);
            std::process::exit(1);
        }
    }
});


pub fn get_mutable_model<'a>() -> MutexGuard<'a, dnn::Net> {
    MODEL.lock().unwrap()
}

pub fn init_dnn() {

    if get_cuda_enabled_device_count().unwrap() >= 1 {
        println!("CUDA is enabled!");
    }

    let mut net = get_mutable_model();

    net.set_preferable_backend(dnn::DNN_BACKEND_CUDA).unwrap();
    net.set_preferable_target(dnn::DNN_TARGET_CUDA).unwrap();


    
}
