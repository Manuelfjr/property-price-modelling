from  pandas import DataFrame


def ranking_output(
        args: list
        ) -> DataFrame:
    data_rank = DataFrame(args, 
                            index = ["y_true", "y_pred"]).T
    data_rank = data_rank.sort_values("y_true", ascending = False)
    data_rank["rank_true"] = (
        data_rank
        .sort_values("y_true", ascending = False)
        .reset_index(drop=True)
        .index
    )
    data_rank = data_rank.sort_values("y_pred", ascending = False)
    data_rank["rank_pred"] = (
        data_rank
        .sort_values("y_pred", ascending = False)
        .reset_index(drop=True)
        .index
    )
    data_rank = data_rank.sample(frac=1).reset_index(drop = True)

    return data_rank