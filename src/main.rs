#[macro_use]
extern crate log;
extern crate env_logger;
extern crate rulinalg;

mod gaussian;

use rulinalg::matrix::{Matrix, BaseMatrix};
use rulinalg::utils;

fn main() {
    env_logger::init().unwrap();

    let train = Matrix::new(20,
                            2,
                            vec![8.692, 11.023,
                                 8.349, 4.634,
                                 10.644, 5.380,
                                 9.680, 7.341,
                                 8.203, 5.507,
                                 7.681, 5.459,
                                 6.453, 5.671,
                                 7.039, 5.663,
                                 8.568, 9.348,
                                 9.551, 7.882,
                                 6.845, 5.609,
                                 6.564, 7.641,
                                 7.459, 6.360,
                                 4.832, 8.555,
                                 5.148, 6.509,
                                 7.292, 5.393,
                                 8.928, 8.124,
                                 7.738, 9.005,
                                 5.434, 8.383,
                                 8.559, 6.835]);
    debug!("train data:\n{}", train);
    let class = Matrix::new(20,
                            2,
                            vec![1.0, 0.0,
                                 1.0, 0.0,
                                 1.0, 0.0,
                                 1.0, 0.0,
                                 1.0, 0.0,
                                 1.0, 0.0,
                                 1.0, 0.0,
                                 1.0, 0.0,
                                 1.0, 0.0,
                                 1.0, 0.0,
                                 0.0, 1.0,
                                 0.0, 1.0,
                                 0.0, 1.0,
                                 0.0, 1.0,
                                 0.0, 1.0,
                                 0.0, 1.0,
                                 0.0, 1.0,
                                 0.0, 1.0,
                                 0.0, 1.0,
                                 0.0, 1.0]);
    debug!("class data:\n{}", class);
    let targets = Matrix::new(21,
                              2,
                              vec![8.0, 8.0,
                                   8.692, 11.023,
                                   8.349, 4.634,
                                   10.644, 5.380,
                                   9.680, 7.341,
                                   8.203, 5.507,
                                   7.681, 5.459,
                                   6.453, 5.671,
                                   7.039, 5.663,
                                   8.568, 9.348,
                                   9.551, 7.882,
                                   6.845, 5.609,
                                   6.564, 7.641,
                                   7.459, 6.360,
                                   4.832, 8.555,
                                   5.148, 6.509,
                                   7.292, 5.393,
                                   8.928, 8.124,
                                   7.738, 9.005,
                                   5.434, 8.383,
                                   8.559, 6.835]);
    debug!("targets data:\n{}", targets);

    let class_count = class.cols();
    let feature_count = train.cols();
    let data_counts = train.rows() as f64;

    let mut class_counts = vec![0; class_count];
    let mut class_data: Vec<Vec<usize>> = vec![Vec::new(); class_count];

    let mut g = gaussian::Gaussian::from_model(class_count, feature_count);

    // Detect rows of the train data, that match specific class.
    for (num_row, row) in class.iter_rows().enumerate() {
        let mut class: i32 = -1;
        for (num_col, c) in row.into_iter().enumerate() {
            if *c == 1f64 {
                class = num_col as i32;
            }
        }

        // If class not detect then will be panic because index is -1.
        class_data[class as usize].push(num_row);
        class_counts[class as usize] += 1;
    }
    debug!("class_data: {:?}", class_data);
    debug!("class_counts: {:?}", class_counts);

    // Compute Gaussian function for every class and feature.
    for (num_class, rows_of_inputs) in class_data.into_iter().enumerate() {
        g.compute_gaussian(&train.select_rows(&rows_of_inputs), num_class);
    }

    debug!("Computed Gaussian: {:?}", g);

    // Compute Class Prior Probability (P(c)) for classes.
    let class_prior: Vec<f64> = class_counts.iter().map(|c| *c as f64 / data_counts).collect();
    debug!("Class Prior Probability: {:?}", class_prior);

    let result = g.compute_likehood_and_predict(&targets, &class_prior);
    debug!("Result:\n{}", result);
    for (target_num, row) in result.iter_rows().enumerate() {
        println!("Class of target #{} is {}",
                 target_num,
                 utils::argmax(row).0 + 1);
    }
}

#[cfg(test)]
mod tests {
    use gaussian;
    use rulinalg::matrix::{Matrix, BaseMatrix};
    use rulinalg::utils;

    #[test]
    fn test_gaussian() {
        let train = Matrix::new(6,
                                2,
                                vec![1.0, 1.1, 1.1, 0.9, 2.2, 2.3, 2.5, 2.7, 5.2, 4.3, 6.2, 7.3]);

        let class = Matrix::new(6,
                                3,
                                vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
                                     0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
        let targets = Matrix::new(6,
                                  2,
                                  vec![1.0, 1.1, 1.1, 0.9, 2.2, 2.3, 2.5, 2.7, 5.2, 4.3, 6.2, 7.3]);

        let class_count = class.cols();
        let feature_count = train.cols();
        let data_counts = train.rows() as f64;

        let mut class_counts = vec![0; class_count];
        let mut class_data: Vec<Vec<usize>> = vec![Vec::new(); class_count];

        let mut g = gaussian::Gaussian::from_model(class_count, feature_count);

        // Detect rows of the train data, that match specific class.
        for (num_row, row) in class.iter_rows().enumerate() {
            let mut class: i32 = -1;
            for (num_col, c) in row.into_iter().enumerate() {
                if *c == 1f64 {
                    class = num_col as i32;
                }
            }

            // If class not detect then will be panic because index is -1.
            class_data[class as usize].push(num_row);
            class_counts[class as usize] += 1;
        }
        debug!("class_data: {:?}", class_data);
        debug!("class_counts: {:?}", class_counts);

        // Compute Gaussian function for every class and feature.
        for (num_class, rows_of_inputs) in class_data.into_iter().enumerate() {
            g.compute_gaussian(&train.select_rows(&rows_of_inputs), num_class);
        }

        debug!("Computed Gaussian: {:?}", g);

        // Compute Class Prior Probability (P(c)) for classes.
        let class_prior: Vec<f64> = class_counts.iter().map(|c| *c as f64 / data_counts).collect();
        debug!("Class Prior Probability: {:?}", class_prior);

        let result = g.compute_likehood_and_predict(&targets, &class_prior);
        debug!("Result:\n{}", result);
        let outputs: Vec<usize> = result.iter_rows().map(|x| utils::argmax(x).0).collect();

        assert_eq!(outputs, [0, 0, 1, 1, 2, 2]);
    }
}
