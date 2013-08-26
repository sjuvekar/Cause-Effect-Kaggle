import transformed_features as tf
import gaussian_process as gp
import anm
import independent_component as indcomp
import numpy as np
import scipy
import hsic
import features as f

def feature_extractor():
    features = [('Number of Samples', 'A', f.SimpleTransform(transformer=len)),
                ('A: Number of Unique Samples', 'A', f.SimpleTransform(transformer=f.count_unique)),
                ('B: Number of Unique Samples', 'B', f.SimpleTransform(transformer=f.count_unique)),
                ('A: Mean', 'A', f.SimpleTransform(transformer=np.mean)),
                ('B: Mean', 'B', f.SimpleTransform(transformer=np.mean)),
                ('A: Max', 'A', f.SimpleTransform(transformer=np.max)),
                ('B: Max', 'B', f.SimpleTransform(transformer=np.max)),
                ('A: Min', 'A', f.SimpleTransform(transformer=np.min)),
                ('B: Min', 'B', f.SimpleTransform(transformer=np.min)),
                ('A: Range', 'A', f.SimpleTransform(transformer=f.rng)),
                ('B: Range', 'B', f.SimpleTransform(transformer=f.rng)),
                ('A: Median', 'A', f.SimpleTransform(transformer=f.median)),
                ('B: Median', 'B', f.SimpleTransform(transformer=f.median)),
                ('A: Percentile 25', 'A', f.SimpleTransform(transformer=f.percentile25)),
                ('B: Percentile 25', 'B', f.SimpleTransform(transformer=f.percentile25)),
                ('A: Percentile 75', 'A', f.SimpleTransform(transformer=f.percentile75)),
                ('B: Percentile 75', 'B', f.SimpleTransform(transformer=f.percentile75)),
                ('A: Bollinger', 'A', f.SimpleTransform(transformer=f.bollinger)),
                ('B: Bollinger', 'B', f.SimpleTransform(transformer=f.bollinger)),
                ('A: Std', 'A', f.SimpleTransform(transformer=np.std)),
                ('B: Std', 'B', f.SimpleTransform(transformer=np.std)),
                ('A: Variation', 'A', f.SimpleTransform(transformer=f.sharpe)),
                ('B: Variation', 'B', f.SimpleTransform(transformer=f.sharpe)),
                ('A: Skew', 'A', f.SimpleTransform(transformer=scipy.stats.skew)),
                ('B: Skew', 'B', f.SimpleTransform(transformer=scipy.stats.skew)),
                ('A: Kurtosis', 'A', f.SimpleTransform(transformer=scipy.stats.kurtosis)),
                ('B: Kurtosis', 'B', f.SimpleTransform(transformer=scipy.stats.kurtosis)),
                #('A: Normalized Entropy', 'A', f.SimpleTransform(transformer=f.normalized_entropy)),
                #('B: Normalized Entropy', 'B', f.SimpleTransform(transformer=f.normalized_entropy)),
                ('linregress', ['A','B'], f.MultiColumnTransform(f.linregress)),
                ('linregress_rev', ['B','A'], f.MultiColumnTransform(f.linregress)),
                ('complex_regress', ['A', 'B'], f.MultiColumnTransform(tf.complex_regress)),
                ('complex_regress rev', ['B', 'A'], f.MultiColumnTransform(tf.complex_regress)),
                #('gaussian process', ['A', 'B'], f.MultiColumnTransform(gp.gaussian_fit_likelihood)),
                #('gaussian process rev', ['B', 'A'], f.MultiColumnTransform(gp.gaussian_fit_likelihood)),
                ('anm', ['A', 'B'], f.MultiColumnTransform(anm.anm_fit)),
                ('anm rev', ['B', 'A'], f.MultiColumnTransform(anm.anm_fit)),
                ('Independent Components', ['A', 'B'], f.MultiColumnTransform(indcomp.independent_component)),
                ('ttest_ind', ['A','B'], f.MultiColumnTransform(f.ttest_ind)),
                ('ks_2samp', ['A','B'], f.MultiColumnTransform(f.ks_2samp)),
                ('kruskal', ['A','B'], f.MultiColumnTransform(f.kruskal)),
                ('bartlett', ['A', 'B'], f.MultiColumnTransform(f.bartlett)),
                ('levene', ['A','B'], f.MultiColumnTransform(f.levene)),
                ('A: shapiro', 'A', f.SimpleTransform(transformer=f.shapiro)),
                ('B: shapiro', 'B', f.SimpleTransform(transformer=f.shapiro)),
                ('fligner', ['A','B'], f.MultiColumnTransform(f.fligner)),
                ('mood', ['A','B'], f.MultiColumnTransform(f.mood)),
                ('oneway', ['A','B'], f.MultiColumnTransform(f.oneway)),
                ('Pearson R', ['A','B'], f.MultiColumnTransform(f.correlation)),
                ('Pearson R Magnitude', ['A','B'], f.MultiColumnTransform(f.correlation_magnitude)),
                ('Entropy Difference', ['A','B'], f.MultiColumnTransform(f.entropy_difference)),
                ('Adjusted Mutual Information', ['A', 'B'], f.MultiColumnTransform(tf.adjusted_mutual_information)),
                ('Adjusted Rand', ['A', 'B'], f.MultiColumnTransform(tf.adjusted_rand)),
                ('Mutual Information', ['A', 'B'], f.MultiColumnTransform(tf.mutual_information)),
                ('Homogeneity Completeness', ['A', 'B'], f.MultiColumnTransform(tf.homogeneity_completeness)),
                ('Normalized Mutual Info', ['A', 'B'], f.MultiColumnTransform(tf.normalized_mutual_information)),
                #('braycurtis', ['A','B'], f.MultiColumnTransform(f.braycurtis)),
                #('canberra', ['A','B'], f.MultiColumnTransform(f.canberra)),
                #('chebyshev', ['A','B'], f.MultiColumnTransform(f.chebyshev)),
                #('cityblock', ['A','B'], f.MultiColumnTransform(f.cityblock)),
                #('cosine', ['A','B'], f.MultiColumnTransform(f.cosine)),
                #('hamming', ['A','B'], f.MultiColumnTransform(f.hamming)),
                #('minkowski', ['A','B'], f.MultiColumnTransform(f.minkowski)),
                #('sqeuclidean', ['A','B'], f.MultiColumnTransform(f.sqeuclidean)),
                #('HSIC', ['A', 'B'], f.MultiColumnTransform(hsic.hsic_score)),
                ]

    combined = f.FeatureMapper(features)
    return combined


