import h2o
from h2o.estimators.deeplearning import H2ODeepLearningEstimator

# Connect to the local H2O server
h2o.init(ip="127.0.0.1", port="54321")
# In case cluster was already running
h2o.remove_all()

# Loading data sets from local storage [csv file]
training_frame = h2o.import_file(path="RUL-H20 Complete Train.csv")
testing_frame = h2o.import_file(path="RUL-H20 Complete Test.csv")

# Summary of data
# training_frame.describe()
# testing_frame.describe()

# Specify predicting and response variables
training_variables = training_frame.names[2:-1]
response_variable = training_frame.names[-1]

# Help on setting up parameters in H2ODeepLearningEstimator
'''
Parameters
 |  ----------
 |    model_id : str
 |      Destination id for this model; auto-generated if not specified.
 |
 |    training_frame : str
 |      Id of the training data frame (Not required, to allow initial validation of model parameters).
 |
 |    validation_frame : str
 |      Id of the validation data frame.
 |
 |    nfolds : int
 |      Number of folds for N-fold cross-validation (0 to disable or >= 2).
 |      Default: 0
 |
 |    keep_cross_validation_predictions : bool
 |      Whether to keep the predictions of the cross-validation models.
 |      Default: False
 |
 |    keep_cross_validation_fold_assignment : bool
 |      Whether to keep the cross-validation fold assignment.
 |      Default: False
 |
 |    fold_assignment : "AUTO" | "Random" | "Modulo" | "Stratified"
 |      Cross-validation fold assignment scheme, if fold_column is not specified. The 'Stratified' option will stratify
 |      the folds based on the response variable, for classification problems.
 |      Default: "AUTO"
 |
 |    fold_column : VecSpecifier
 |      Column with cross-validation fold index assignment per observation.
 |
 |    response_column : VecSpecifier
 |      Response variable column.
 |
 |    ignored_columns : list(str)
 |      Names of columns to ignore for training.
 |
 |    ignore_const_cols : bool
 |      Ignore constant columns.
 |      Default: True
 |
 |    score_each_iteration : bool
 |      Whether to score during each iteration of model training.
 |      Default: False
 |
 |    weights_column : VecSpecifier
 |      Column with observation weights. Giving some observation a weight of zero is equivalent to excluding it from the
 |      dataset; giving an observation a relative weight of 2 is equivalent to repeating that row twice. Negative
 |      weights are not allowed.
 |
 |    offset_column : VecSpecifier
 |      Offset column. This will be added to the combination of columns before applying the link function.
 |
 |    balance_classes : bool
 |      Balance training data class counts via over/under-sampling (for imbalanced data).
 |      Default: False
 |
 |    class_sampling_factors : list(float)
 |      Desired over/under-sampling ratios per class (in lexicographic order). If not specified, sampling factors will
 |      be automatically computed to obtain class balance during training. Requires balance_classes.
 |
 |    max_after_balance_size : float
 |      Maximum relative size of the training data after balancing class counts (can be less than 1.0). Requires
 |      balance_classes.
 |      Default: 5.0
 |
 |    max_confusion_matrix_size : int
 |      Maximum size (# classes) for confusion matrices to be printed in the Logs.
 |      Default: 20
 |
 |    max_hit_ratio_k : int
 |      Max. number (top K) of predictions to use for hit ratio computation (for multi-class only, 0 to disable).
 |      Default: 0
 |
 |    checkpoint : str
 |      Model checkpoint to resume training with.
 |
 |    pretrained_autoencoder : str
 |      Pretrained autoencoder model to initialize this model with.
 |
 |    overwrite_with_best_model : bool
 |      If enabled, override the final model with the best model found during training.
 |      Default: True
 |
 |    use_all_factor_levels : bool
 |      Use all factor levels of categorical variables. Otherwise, the first factor level is omitted (without loss of
 |      accuracy). Useful for variable importances and auto-enabled for autoencoder.
 |      Default: True
 |
 |    standardize : bool
 |      If enabled, automatically standardize the data. If disabled, the user must provide properly scaled input data.
 |      Default: True
 |
 |    activation : "Tanh" | "TanhWithDropout" | "Rectifier" | "RectifierWithDropout" | "Maxout" | "MaxoutWithDropout"
 |      Activation function.
 |      Default: "Rectifier"
 |
 |    hidden : list(int)
 |      Hidden layer sizes (e.g. [100, 100]).
 |      Default: [200, 200]
 |
 |    epochs : float
 |      How many times the dataset should be iterated (streamed), can be fractional.
 |      Default: 10.0
 |
 |    train_samples_per_iteration : int
 |      Number of training samples (globally) per MapReduce iteration. Special values are 0: one epoch, -1: all
 |      available data (e.g., replicated training data), -2: automatic.
 |      Default: -2
 |
 |    target_ratio_comm_to_comp : float
 |      Target ratio of communication overhead to computation. Only for multi-node operation and
 |      train_samples_per_iteration = -2 (auto-tuning).
 |      Default: 0.05
 |
 |    seed : int
 |      Seed for random numbers (affects sampling) - Note: only reproducible when running single threaded.
 |      Default: -1
 |
 |    adaptive_rate : bool
 |      Adaptive learning rate.
 |      Default: True
 |
 |    rho : float
 |      Adaptive learning rate time decay factor (similarity to prior updates).
 |      Default: 0.99
 |
 |    epsilon : float
 |      Adaptive learning rate smoothing factor (to avoid divisions by zero and allow progress).
 |      Default: 1e-08
 |
 |    rate : float
 |      Learning rate (higher => less stable, lower => slower convergence).
 |      Default: 0.005
 |
 |    rate_annealing : float
 |      Learning rate annealing: rate / (1 + rate_annealing * samples).
 |      Default: 1e-06
 |
 |    rate_decay : float
 |      Learning rate decay factor between layers (N-th layer: rate * rate_decay ^ (n - 1).
 |      Default: 1.0
 |
 |    momentum_start : float
 |      Initial momentum at the beginning of training (try 0.5).
 |      Default: 0.0
 |
 |    momentum_ramp : float
 |      Number of training samples for which momentum increases.
 |      Default: 1000000.0
 |
 |    momentum_stable : float
 |      Final momentum after the ramp is over (try 0.99).
 |      Default: 0.0
 |
 |    nesterov_accelerated_gradient : bool
 |      Use Nesterov accelerated gradient (recommended).
 |      Default: True
 |
 |    input_dropout_ratio : float
 |      Input layer dropout ratio (can improve generalization, try 0.1 or 0.2).
 |      Default: 0.0
 |
 |    hidden_dropout_ratios : list(float)
 |      Hidden layer dropout ratios (can improve generalization), specify one value per hidden layer, defaults to 0.5.
 |
 |    l1 : float
 |      L1 regularization (can add stability and improve generalization, causes many weights to become 0).
 |      Default: 0.0
 |
 |    l2 : float
 |      L2 regularization (can add stability and improve generalization, causes many weights to be small.
 |      Default: 0.0
 |
 |    max_w2 : float
 |      Constraint for squared sum of incoming weights per unit (e.g. for Rectifier).
 |      Default: infinity
 |
 |    initial_weight_distribution : "UniformAdaptive" | "Uniform" | "Normal"
 |      Initial weight distribution.
 |      Default: "UniformAdaptive"
 |
 |    initial_weight_scale : float
 |      Uniform: -value...value, Normal: stddev.
 |      Default: 1.0
 |
 |    initial_weights : list(str)
 |      A list of H2OFrame ids to initialize the weight matrices of this model with.
 |
 |    initial_biases : list(str)
 |      A list of H2OFrame ids to initialize the bias vectors of this model with.
 |
 |    loss : "Automatic" | "CrossEntropy" | "Quadratic" | "Huber" | "Absolute" | "Quantile"
 |      Loss function.
 |      Default: "Automatic"
 |
 |    distribution : "AUTO" | "bernoulli" | "multinomial" | "gaussian" | "poisson" | "gamma" | "tweedie" | "laplace" |
 |                   "quantile" | "huber"
 |      Distribution function
 |      Default: "AUTO"
 |
 |    quantile_alpha : float
 |      Desired quantile for Quantile regression, must be between 0 and 1.
 |      Default: 0.5
 |
 |    tweedie_power : float
 |      Tweedie power for Tweedie regression, must be between 1 and 2.
 |      Default: 1.5
 |
 |    huber_alpha : float
 |      Desired quantile for Huber/M-regression (threshold between quadratic and linear loss, must be between 0 and 1).
 |      Default: 0.9
 |
 |    score_interval : float
 |      Shortest time interval (in seconds) between model scoring.
 |      Default: 5.0
 |
 |    score_training_samples : int
 |      Number of training set samples for scoring (0 for all).
 |      Default: 10000
 |
 |    score_validation_samples : int
 |      Number of validation set samples for scoring (0 for all).
 |      Default: 0
 |
 |    score_duty_cycle : float
 |      Maximum duty cycle fraction for scoring (lower: more training, higher: more scoring).
 |      Default: 0.1
 |
 |    classification_stop : float
 |      Stopping criterion for classification error fraction on training data (-1 to disable).
 |      Default: 0.0
 |
 |    regression_stop : float
 |      Stopping criterion for regression error (MSE) on training data (-1 to disable).
 |      Default: 1e-06
 |
 |    stopping_rounds : int
 |      Early stopping based on convergence of stopping_metric. Stop if simple moving average of length k of the
 |      stopping_metric does not improve for k:=stopping_rounds scoring events (0 to disable)
 |      Default: 5
 |
 |    stopping_metric : "AUTO" | "deviance" | "logloss" | "MSE" | "AUC" | "lift_top_group" | "r2" | "misclassification"
 |                      | "mean_per_class_error"
 |      Metric to use for early stopping (AUTO: logloss for classification, deviance for regression)
 |      Default: "AUTO"
 |
 |    stopping_tolerance : float
 |      Relative tolerance for metric-based stopping criterion (stop if relative improvement is not at least this much)
 |      Default: 0.0
 |
 |    max_runtime_secs : float
 |      Maximum allowed runtime in seconds for model training. Use 0 to disable.
 |      Default: 0.0
 |
 |    score_validation_sampling : "Uniform" | "Stratified"
 |      Method used to sample validation dataset for scoring.
 |      Default: "Uniform"
 |
 |    diagnostics : bool
 |      Enable diagnostics for hidden layers.
 |      Default: True
 |
 |    fast_mode : bool
 |      Enable fast mode (minor approximation in back-propagation).
 |      Default: True
 |
 |    force_load_balance : bool
 |      Force extra load balancing to increase training speed for small datasets (to keep all cores busy).
 |      Default: True
 |
 |    variable_importances : bool
 |      Compute variable importances for input features (Gedeon method) - can be slow for large networks.
 |      Default: False
 |
 |    replicate_training_data : bool
 |      Replicate the entire training dataset onto every node for faster training on small datasets.
 |      Default: True
 |
 |    single_node_mode : bool
 |      Run on a single node for fine-tuning of model parameters.
 |      Default: False
 |
 |    shuffle_training_data : bool
 |      Enable shuffling of training data (recommended if training data is replicated and train_samples_per_iteration is
 |      close to #nodes x #rows, of if using balance_classes).
 |      Default: False
 |
 |    missing_values_handling : "Skip" | "MeanImputation"
 |      Handling of missing values. Either Skip or MeanImputation.
 |      Default: "MeanImputation"
 |
 |    quiet_mode : bool
 |      Enable quiet mode for less output to standard output.
 |      Default: False
 |
 |    autoencoder : bool
 |      Auto-Encoder.
 |      Default: False
 |
 |    sparse : bool
 |      Sparse data handling (more efficient for data with lots of 0 values).
 |      Default: False
 |
 |    col_major : bool
 |      #DEPRECATED Use a column major weight matrix for input layer. Can speed up forward propagation, but might slow
 |      down backpropagation.
 |      Default: False
 |
 |    average_activation : float
 |      Average activation for sparse auto-encoder. #Experimental
 |      Default: 0.0
 |
 |    sparsity_beta : float
 |      Sparsity regularization. #Experimental
 |      Default: 0.0
 |
 |    max_categorical_features : int
 |      Max. number of categorical features, enforced via hashing. #Experimental
 |      Default: 2147483647
 |
 |    reproducible : bool
 |      Force reproducibility on small data (will be slow - only uses 1 thread).
 |      Default: False
 |
 |    export_weights_and_biases : bool
 |      Whether to export Neural Network weights and biases to H2O Frames.
 |      Default: False
 |
 |    mini_batch_size : int
 |      Mini-batch size (smaller leads to better fit, larger can speed up and generalize better).
 |      Default: 1
 |
 |    categorical_encoding : "AUTO" | "Enum" | "OneHotInternal" | "OneHotExplicit" | "Binary" | "Eigen"
 |      Encoding scheme for categorical features
 |      Default: "AUTO"
 |
 |    elastic_averaging : bool
 |      Elastic averaging between compute nodes can improve distributed model convergence. #Experimental
 |      Default: False
 |
 |    elastic_averaging_moving_rate : float
 |      Elastic averaging moving rate (only if elastic averaging is enabled).
 |      Default: 0.9
 |
 |    elastic_averaging_regularization : float
 |      Elastic averaging regularization strength (only if elastic averaging is enabled).
 |      Default: 0.001
'''
model = H2ODeepLearningEstimator(hidden=[1000, 1000, 1000], score_each_iteration=True, variable_importances=True)
model.show()

# Help on setting up parameters in train
'''
train(self, x=None, y=None, training_frame=None, offset_column=None, fold_column=None, weights_column=None, validation_frame=None, max_runtime_secs=None, **params)
 |      Train the H2O model.
 |
 |      Parameters
 |      ----------
 |      x : list, None
 |          A list of column names or indices indicating the predictor columns.
 |
 |      y : str, int
 |          An index or a column name indicating the response column.
 |
 |      training_frame : H2OFrame
 |          The H2OFrame having the columns indicated by x and y (as well as any
 |          additional columns specified by fold, offset, and weights).
 |
 |      offset_column : str, optional
 |          The name or index of the column in training_frame that holds the offsets.
 |
 |      fold_column : str, optional
 |          The name or index of the column in training_frame that holds the per-row fold
 |          assignments.
 |
 |      weights_column : str, optional
 |          The name or index of the column in training_frame that holds the per-row weights.
 |
 |      validation_frame : H2OFrame, optional
 |          H2OFrame with validation data to be scored on while training.
 |
 |      max_runtime_secs : float
 |          Maximum allowed runtime in seconds for model training. Use 0 to disable.
'''

model.train(x=training_variables, y=response_variable, training_frame=training_frame)

# Help on setting up parameters in model_performance
'''
 |  model_performance(self, test_data=None, train=False, valid=False, xval=False)
 |      Generate model metrics for this model on test_data.
 |
 |      Parameters
 |      ----------
 |      test_data: H2OFrame, optional
 |        Data set for which model metrics shall be computed against. All three of train, valid and xval arguments are
 |        ignored if test_data is not None.
 |      train: boolean, optional
 |        Report the training metrics for the model.
 |      valid: boolean, optional
 |        Report the validation metrics for the model.
 |      xval: boolean, optional
 |        Report the cross-validation metrics for the model. If train and valid are True, then it defaults to True.
 |
 |      :returns: An object of class H2OModelMetrics.
'''
performance = model.model_performance(test_data=testing_frame)
performance.show()

# Predictions on testing data
predictions = model.predict(testing_frame)
predictions.describe()

