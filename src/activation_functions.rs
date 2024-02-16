#![allow(unused)]

use crate::linear_algebra::Matrix;


//----------------------
// Activation Functions:
//----------------------

pub type ActivationFun = fn(&Matrix) -> Matrix;

pub fn sigmoid(m: &Matrix) -> Matrix {
    let ones = &Matrix::fill_with(1.0f32, m.rows, m.cols);
    let denominator = ones + &Matrix::exp(&Matrix::apply_scalar(&m, -1.0f32));
    ones / &denominator
}

/**
 * Rectified Linear
 */
pub fn relu(m: &Matrix) -> Matrix {
    Matrix::max_of(0.0f32, m)
}

pub fn tanh(m: &Matrix) -> Matrix {
    let mut result = m.clone();

    for x in 0..m.rows {
        for y in 0..m.cols {
            result[x][y] = f32::tanh(m[x][y]);
        }
    }
    result 
}

//---------------------------------
// Activation Derivative Functions:
//---------------------------------

pub type ActivationFunDerivative = fn(&Matrix) -> Matrix;

pub fn d_sigmoid(m: &Matrix) -> Matrix {
    let ones = Matrix::fill_with(1.0f32, m.rows, m.cols);
    let sigmoid = &sigmoid(m);
    sigmoid * &(&ones - sigmoid)
}

pub fn d_relu(m: &Matrix) -> Matrix {
    let mut result = m.clone();

    for x in 0..m.rows {
        for y in 0..m.cols {
            if m[x][y] > 0.0f32 {
                result[x][y] = 1.0f32;
            }
        }
    }
    result 
}

pub fn d_tanh(m: &Matrix) -> Matrix {
    let mut result = m.clone();

    for x in 0..m.rows {
        for y in 0..m.cols {
            result[x][y] = 1.0f32 - f32::powf(f32::tanh(m[x][y]), 2.0f32);
        }
    }
    result 
}