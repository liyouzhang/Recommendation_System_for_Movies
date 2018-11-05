import sys
import pandas as pd
from performotron import Comparer

class RecComparer(Comparer):
    def score(self, predictions):
        """Look at 5% of most highly predicted jokes for each user.
        Return the average actual rating of those jokes.
        """

        df = pd.concat([predictions,
                        self.target], axis=1)

        g = df.groupby('user')

        top_5 = g.rating.transform(
            lambda x: x >= x.quantile(.95)
        )

        return self.target[top_5==1].mean()

if __name__ == "__main__":
    sample_sub_fname = sys.argv[1]
    test=pd.read_csv('data/do_not_use/testing.csv')
    test.rating.name='test_rating'
    rc = RecComparer(test.rating, config_file='config.yaml')

    sample_sub = pd.read_csv(sample_sub_fname)
    if sample_sub.shape[0] != 200209:
        print(" ".join(["Your matrix of predictions is the wrong size.",
        "It should provide ratings",
        " for 200209 entries (yours={}).".format(sample_sub.shape[0])]))
    else:
        rc.report_to_slack(sample_sub)
