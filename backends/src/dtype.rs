use std::fmt;

#[cfg(feature = "clap")]
use clap::ValueEnum;

#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "clap", derive(Clone, ValueEnum))]
pub enum DType {
    // Float16 is not available on accelerate
    #[cfg(any(
        feature = "python",
        all(feature = "candle", not(feature = "accelerate"))
    ))]
    Float16,
    // Bfloat16 is available only when the "python" feature is enabled
    #[cfg(feature = "python")]
    Bfloat16,
    // Float32 is not available on candle cuda
    #[cfg(any(feature = "python", feature = "candle"))]
    Float32,
    // #[cfg(feature = "candle")]
    // Q6K,
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            // Float16 is not available on accelerate
            #[cfg(any(
                feature = "python",
                all(feature = "candle", not(feature = "accelerate"))
            ))]
            DType::Float16 => write!(f, "float16"),

            // Bfloat16 is only available with the "python" feature
            #[cfg(feature = "python")]
            DType::Bfloat16 => write!(f, "bfloat16"),

            // Float32 is not available on candle cuda
            #[cfg(any(feature = "python", feature = "candle"))]
            DType::Float32 => write!(f, "float32"),

            // #[cfg(feature = "candle")]
            // DType::Q6K => write!(f, "q6k"),
        }
    }
}
