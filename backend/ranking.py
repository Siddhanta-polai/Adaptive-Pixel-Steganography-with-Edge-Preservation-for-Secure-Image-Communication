import numpy as np

class AlgorithmRanker:
    CRITERIA = {
        "psnr": {"weight": 0.25, "beneficial": True},
        "ssim": {"weight": 0.25, "beneficial": True},
        "epi": {"weight": 0.15, "beneficial": True},
        "snr": {"weight": 0.15, "beneficial": True},
        "mse": {"weight": 0.10, "beneficial": False},
        "fdm": {"weight": 0.10, "beneficial": False}
    }

    @classmethod
    def rank_algorithms(cls, leaderboard):
        if not leaderboard:
            return []
        names = [item["name"] for item in leaderboard]
        metrics = []
        for item in leaderboard:
            row = [float(item.get(c, 0)) for c in cls.CRITERIA]
            metrics.append(row)
        matrix = np.array(metrics)
        norm_matrix = matrix / np.sqrt(np.sum(matrix**2, axis=0) + 1e-12)
        weights = np.array([cls.CRITERIA[c]["weight"] for c in cls.CRITERIA])
        weighted_matrix = norm_matrix * weights
        ideal_best, ideal_worst = [], []
        for i, crit in enumerate(cls.CRITERIA):
            if cls.CRITERIA[crit]["beneficial"]:
                ideal_best.append(np.max(weighted_matrix[:, i]))
                ideal_worst.append(np.min(weighted_matrix[:, i]))
            else:
                ideal_best.append(np.min(weighted_matrix[:, i]))
                ideal_worst.append(np.max(weighted_matrix[:, i]))
        ideal_best = np.array(ideal_best)
        ideal_worst = np.array(ideal_worst)
        dist_best = np.sqrt(np.sum((weighted_matrix - ideal_best)**2, axis=1))
        dist_worst = np.sqrt(np.sum((weighted_matrix - ideal_worst)**2, axis=1))
        scores = dist_worst / (dist_best + dist_worst + 1e-12)
        ranked = []
        for i, name in enumerate(names):
            ranked.append({
                "name": name,
                "score": round(scores[i], 4),
                "is_ours": leaderboard[i].get("is_ours", False)
            })
        ranked.sort(key=lambda x: x["score"], reverse=True)
        for idx, item in enumerate(ranked):
            item["rank"] = idx + 1
            item["recommended"] = (idx == 0)
        return ranked

    @classmethod
    def get_best_algorithm(cls, leaderboard):
        ranked = cls.rank_algorithms(leaderboard)
        if not ranked:
            return None, "No algorithms to evaluate."
        best = ranked[0]
        explanation = f"🏆 **{best['name']}** is recommended based on PSNR, SSIM, EPI, SNR, MSE, and FDM trade-off."
        return best["name"], explanation