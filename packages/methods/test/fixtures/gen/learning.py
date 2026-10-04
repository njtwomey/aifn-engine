"""Golden values for aifn-compute/classify, from scikit-learn's classifiers on small fixed data."""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def cases() -> dict[str, object]:
    rng = np.random.default_rng(7)
    centres = np.array([[0.0, 0.0], [2.5, 0.5], [1.0, 2.5]])
    x3 = np.vstack([c + rng.normal(scale=0.8, size=(20, 2)) for c in centres])
    y3 = np.repeat([0, 1, 2], 20)
    xq = rng.normal(loc=1.0, scale=1.5, size=(12, 2))

    x2 = x3[:40]
    y2 = y3[:40]

    knn = {
        w: KNeighborsClassifier(n_neighbors=5, weights=w).fit(x3, y3).predict_proba(xq) for w in ("uniform", "distance")
    }
    knn_manhattan = KNeighborsClassifier(n_neighbors=4, p=1).fit(x3, y3).predict_proba(xq)

    counts = rng.poisson(2.0, size=(30, 5)).astype(float)
    ycounts = rng.integers(0, 3, size=30)
    counts[ycounts == 1, 0] += 3
    counts[ycounts == 2, 4] += 3

    tree = DecisionTreeClassifier(random_state=0).fit(x3, y3)
    tree_entropy = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=0).fit(x3, y3)
    path = tree.cost_complexity_pruning_path(x3, y3)
    pruned = DecisionTreeClassifier(random_state=0, ccp_alpha=0.02).fit(x3, y3)
    yreg = np.sin(x3[:, 0]) + 0.3 * x3[:, 1] + rng.normal(scale=0.1, size=60)
    rtree = DecisionTreeRegressor(max_depth=3, random_state=0).fit(x3, yreg)

    def tree_json(t: DecisionTreeClassifier | DecisionTreeRegressor) -> dict[str, object]:
        s = t.tree_
        return {
            "feature": s.feature,
            "threshold": s.threshold,
            "left": s.children_left,
            "right": s.children_right,
            "impurity": s.impurity,
            "count": s.n_node_samples,
            "importances": t.feature_importances_,
        }

    # sklearn ships no type stubs, so Pyright takes a parameter's type from its default (gamma="scale", dual="auto",
    # tol=1e-3); the ignores below mark those arguments.
    svc = SVC(C=1.0, kernel="rbf", gamma=0.5, tol=1e-6).fit(x2, y2)  # pyright: ignore[reportArgumentType]
    svc_linear = SVC(C=0.5, kernel="linear", tol=1e-6).fit(x2, y2)
    lsvc = LinearSVC(C=0.5, loss="hinge", dual=True, tol=1e-10, max_iter=1_000_000, random_state=0).fit(x2, y2)  # pyright: ignore[reportArgumentType]
    cs = LinearSVC(C=0.5, multi_class="crammer_singer", tol=1e-10, max_iter=1_000_000, random_state=0).fit(x3, y3)
    perceptron = Perceptron(shuffle=False, eta0=1.0, max_iter=1000, tol=None).fit(x2, y2)  # pyright: ignore[reportArgumentType]

    ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=5, random_state=0).fit(x3, y3)
    gbr = GradientBoostingRegressor(n_estimators=5, max_depth=2, learning_rate=0.3, random_state=0).fit(x3, yreg)
    gbc = GradientBoostingClassifier(n_estimators=5, max_depth=2, learning_rate=0.3, random_state=0).fit(x2, y2)
    gbm = GradientBoostingClassifier(n_estimators=4, max_depth=2, learning_rate=0.3, random_state=0).fit(x3, y3)

    ovr = OneVsRestClassifier(LogisticRegression(C=1.0, tol=1e-12, max_iter=10000)).fit(x3, y3)
    ovo = OneVsOneClassifier(LogisticRegression(C=1.0, tol=1e-12, max_iter=10000)).fit(x3, y3)

    return {
        "x3": x3,
        "y3": y3,
        "xq": xq,
        "knn": knn,
        "knn_manhattan": knn_manhattan,
        "gnb": GaussianNB().fit(x3, y3).predict_proba(xq),
        "counts": counts,
        "ycounts": ycounts,
        "mnb": MultinomialNB(alpha=0.5).fit(counts, ycounts).predict_proba(counts[:6]),
        "bnb": BernoulliNB(alpha=1.0, binarize=1.5).fit(counts, ycounts).predict_proba(counts[:6]),
        "lda": LinearDiscriminantAnalysis().fit(x3, y3).predict_proba(xq),
        "lda_ratio": LinearDiscriminantAnalysis().fit(x3, y3).explained_variance_ratio_,
        "qda": QuadraticDiscriminantAnalysis(reg_param=0.1).fit(x3, y3).predict_proba(xq),
        "tree": tree_json(tree),
        "tree_proba": tree.predict_proba(xq),
        "tree_entropy": tree_json(tree_entropy),
        "path": {"alphas": path.ccp_alphas, "impurities": path.impurities},
        "pruned": tree_json(pruned),
        "yreg": yreg,
        "rtree": tree_json(rtree),
        "rtree_predict": rtree.predict(xq),
        "svc": {"decision": svc.decision_function(xq), "support": svc.support_, "dual": np.asarray(svc.dual_coef_)[0]},
        "svc_linear": {"decision": svc_linear.decision_function(xq), "coef": svc_linear.coef_[0]},
        "lsvc": {"coef": np.asarray(lsvc.coef_)[0], "intercept": np.asarray(lsvc.intercept_)[0]},
        "cs": {"coef": cs.coef_, "intercept": cs.intercept_, "decision": cs.decision_function(xq)},
        "perceptron": {"coef": np.asarray(perceptron.coef_)[0], "intercept": np.asarray(perceptron.intercept_)[0]},
        "ada": {"weights": ada.estimator_weights_, "errors": ada.estimator_errors_, "predict": ada.predict(xq)},
        "gbr": {"staged": np.array(list(gbr.staged_predict(xq)))},
        "gbc": {"decision": gbc.decision_function(xq), "proba": gbc.predict_proba(xq)},
        "gbm": {"proba": gbm.predict_proba(xq)},
        "ovr": {"decision": ovr.decision_function(xq), "proba": ovr.predict_proba(xq)},
        "ovo": {"decision": ovo.decision_function(xq)},
    }
