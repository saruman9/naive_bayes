//! Computes the Gaussian function for matrix of train data.

use rulinalg::matrix::{BaseMatrix, BaseMatrixMut, Matrix, Axes};

use std::f64::consts;
use std::fmt;

/// Contains `mu` - expected value and `sigma ^ 2` - variance for computing Gaussian function:
///
/// p(x == v | c) = 1/sqrt(2 * Pi * sigma_c ^ 2) * e ^ (-(v - mu_c) ^ 2 / 2 * sigma_c ^ 2)
///
/// where p(x == v | c) is P(x | C_k) or likehood.
///
/// `Gaussian` struct view:
///
///         | feature_1 | feature_2 | feature_3 | ... | feature_n |
/// --------+-----------+-----------+-----------+-----+-----------+
/// Class_1 | exp, var  | exp, var  | exp, var  | ... | exp, var  |
/// --------+-----------+-----------+-----------+-----+-----------+
/// Class_2 | exp, var  | exp, var  | exp, var  | ... | exp, var  |
/// --------+-----------+-----------+-----------+-----+-----------+
/// ...     |    ...    |    ...    |    ...    | ... |   ...     |
/// --------+-----------+-----------+-----------+-----+-----------+
/// Class_k | exp, var  | exp, var  | exp, var  | ... | exp, var  |
/// --------+-----------+-----------+-----------+-----+-----------+
///
/// where `exp` - expected value, `var` - variance.
pub struct Gaussian {
    /// `mu` - expected value.
    expected: Matrix<f64>,
    /// `sigma ^ 2` - variance.
    variance: Matrix<f64>,
}

impl Gaussian {
    /// Get expected value (mean) of data.
    pub fn expected(&self) -> &Matrix<f64> {
        &self.expected
    }

    /// Get variance of data.
    pub fn variance(&self) -> &Matrix<f64> {
        &self.variance
    }

    /// Create empty parameters and likehoods for data.
    pub fn from_model(class_count: usize, features_count: usize) -> Self {
        Gaussian {
            expected: Matrix::zeros(class_count, features_count),
            variance: Matrix::zeros(class_count, features_count),
        }
    }

    /// Compute parameters (expected value, variance) for the Class of data.
    pub fn compute_gaussian(&mut self, data: &Matrix<f64>, class_num: usize) {
        let expected = data.mean(Axes::Row).into_vec();
        // Count of rows must be greater than 1.
        let variance = data.variance(Axes::Row).map(|x| x.into_vec()).unwrap_or(vec![0f64]);

        let features: usize = data.cols();

        for (feature_id, (exp, var)) in expected.into_iter().zip(variance.into_iter()).enumerate() {
            self.expected.mut_data()[class_num * features + feature_id] = exp;
            self.variance.mut_data()[class_num * features + feature_id] = var;
        }
        debug!("Compute Gaussian (class: {}):\nexp:\n{}\nvar:\n{}",
               class_num,
               self.expected.select_rows(&[class_num]),
               self.variance.select_rows(&[class_num]));
    }

    /// Compute likehood and Posterior Probability for each class of data.
    pub fn compute_likehood_and_predict(&self,
                                        targets: &Matrix<f64>,
                                        class_prior: &[f64])
                                        -> Matrix<f64> {
        let class_count = self.variance().rows();
        let mut post_prob: Vec<f64> = Vec::with_capacity(targets.rows() * class_count);

        for (num_class, cls_prior) in class_prior.into_iter().enumerate() {
            let first_factor = self.variance()
                .select_rows(&[num_class])
                .apply(&|sigma| 1f64 / (sigma * 2.0 * consts::PI).sqrt());
            debug!("first factor({}):\n{}", num_class, first_factor);
            let second_factor = (targets -
                                 self.expected().select_rows(&vec![num_class; targets.rows()]))
                .apply(&|x| (x * x) * -0.5)
                .elediv(&self.variance().select_rows(&vec![num_class; targets.rows()]))
                .apply(&|x| x.exp());
            debug!("second factor({}):\n{}", num_class, second_factor);
            let likehood =
                second_factor.elemul(&first_factor.select_rows(&vec!(0; targets.rows())));
            debug!("likehood({}):\n{}", num_class, likehood);

            for row in likehood.iter_rows() {
                let product: f64 = row.iter().product();
                post_prob.push(product * cls_prior);
            }
        }

        Matrix::new(class_count, targets.rows(), post_prob).transpose()
    }
}

impl fmt::Debug for Gaussian {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f,
               "Gaussian:\nexp:\n{}\nvar:\n{}",
               self.expected(),
               self.variance())
    }
}
