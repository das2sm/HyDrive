ps aux |grep eval_b2d_multi |grep -v grep | awk '{print $2}' | xargs -r kill -9
ps aux |grep leaderboard_evaluator.py |grep -v grep | awk '{print $2}' | xargs -r kill -9
ps aux |grep CarlaUE4 |grep -v grep | awk '{print $2}' | xargs -r kill -9