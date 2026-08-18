#!/bin/bash
#baseline
./local/st50_dur250_win10_thr50_cd05.sh

#start_window
./local/st0_dur250_win10_thr50_cd05.sh

#threshold
./local/st50_dur250_win10_thr75_cd05.sh
./local/st50_dur250_win10_thr90_cd05.sh

#duration
./local/st50_dur100_win10_thr50_cd05.sh
./local/st50_dur500_win10_thr50_cd05.sh

#window
./local/st50_dur250_win30_thr50_cd05.sh
./local/st50_dur250_win60_thr50_cd05.sh

#no admission control
./local/st50_dur250_noac.sh
./local/st50_dur100_noac.sh
./local/st50_dur500_noac.sh
./local/st0_dur250_noac.sh