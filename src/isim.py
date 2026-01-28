import numpy as np
from scipy.stats import hypergeom
from tqdm import tqdm
from joblib import Parallel, delayed
from utils import pearson, spearman, r2_score

def compute_similarity(x, y, thresholds_x, thresholds_y):
    # Calculate the similarity and p-value of two vectors at different thresholds.
    Stemp = np.zeros((len(thresholds_x), len(thresholds_y)))
    Ptemp = np.ones((len(thresholds_x), len(thresholds_y)))

    for t1, thr_x in enumerate(thresholds_x):
        for t2, thr_y in enumerate(thresholds_y):
            x2 = np.sign(x * (np.abs(x) >= thr_x))
            y2 = np.sign(y * (np.abs(y) >= thr_y))

            num_overlap = np.count_nonzero(x2 * y2 > 0)
            num_x = np.count_nonzero(x2)
            num_y = np.count_nonzero(y2)
            total = len(x2)

            if num_x > 0 and num_y > 0 and num_overlap > 1:
                Ptemp[t1, t2] = hypergeom.sf(num_overlap - 1, total, num_y, num_x)

            Stemp[t1, t2] = num_overlap / max(1, np.count_nonzero(np.abs(x2) + np.abs(y2)))

    min_p_idx = np.unravel_index(np.argmin(Ptemp), Ptemp.shape)
    best_similarity = Stemp[min_p_idx]
    best_p_value = Ptemp[min_p_idx]

    pear = pearson(x, y)
    spear = spearman(x, y)
    r2 = r2_score(x, y)

    return best_similarity, best_p_value, pear, spear, r2


def iSim(Data1z, Data2z=None, Data1p=None, Data2p=None, pthr=0.05, n_jobs=8):
    """
    Calculate pairwise similarity between two data matrices and compute p-values using a hypergeometric distribution.
    Data1z: First set of data (Z-score or fold change)
    Data2z: Second set of data (if empty, calculate similarity within Data1z)
    Data1p, Data2p: p-value matrices (used to filter significant data)
    pthr: p-value threshold
    n_jobs: Number of cores for parallel computation (-1 indicates using all cores)
    """
    Data1z = np.array(Data1z)
    Data1p = np.array(Data1p) if Data1p is not None else np.zeros_like(Data1z)

    if Data2z is None:
        Data2z, Data2p = Data1z, Data1p
        symmetric = True
    else:
        Data2z = np.array(Data2z)
        Data2p = np.array(Data2p) if Data2p is not None else np.zeros_like(Data2z)
        symmetric = False

    Pdata1 = Data1z * (Data1p <= pthr)
    Pdata2 = Data2z * (Data2p <= pthr)

    num_cols1, num_cols2 = Pdata1.shape[1], Pdata2.shape[1]
    Cs = np.zeros((num_cols1, num_cols2))
    Ps = np.ones((num_cols1, num_cols2))
    pearsons = np.zeros((num_cols1, num_cols2))
    spearmans = np.zeros((num_cols1, num_cols2))
    r2 = np.zeros((num_cols1, num_cols2))

    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_similarity)(
            Pdata1[:, i], Pdata2[:, j],
            np.unique(np.maximum(0.5, np.floor(np.abs(Pdata1[:, i]) * 10) / 10)),
            np.unique(np.maximum(0.5, np.floor(np.abs(Pdata2[:, j]) * 10) / 10))
        )
        for i in tqdm(range(num_cols1), desc="Computing Similarity")
        for j in range(i + 1 if symmetric else num_cols2)
    )

    index = 0
    for i in range(num_cols1):
        for j in range(i + 1 if symmetric else num_cols2):
            Cs[i, j], Ps[i, j], pearsons[i, j], spearmans[i, j], r2[i, j] = results[index]
            index += 1

    if symmetric:
        Cs += Cs.T
        Ps += Ps.T

    CsIte = np.abs(Cs) * (-np.log10(Ps))
    A1 = (CsIte - np.nanmean(CsIte, axis=0)) / np.nanstd(CsIte, axis=0)
    A2 = (CsIte - np.nanmean(CsIte, axis=1, keepdims=True)) / np.nanstd(CsIte, axis=1, keepdims=True)
    A1[A1 < 0] = 0
    A2[A2 < 0] = 0
    A = np.sqrt(A1 ** 2 + A2 ** 2) * np.sign(Cs)
    CsIte = CsIte * np.sign(Cs)

    return Cs, Ps, A, CsIte, pearsons, spearmans, r2

if __name__ == '__main__':
    np.random.seed(42)
    Data1z = np.random.randn(1000, 10)
    # Data1p = np.random.rand(1000, 10)

    Cs, Ps, A, CsIte = iSim(Data1z, pthr=0.05)
    print("similarity Cs:\n", Cs)
    print("p value Ps:\n", Ps)
    print("CsIte:\n", CsIte)
    print("A:\n", A)
