## 26 0

    val_loss_every = 125 # every how many steps to evaluate val loss? 0 for only at the end

    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):

        val_loss = 0

                val_loss += model(inputs, targets, get_window_size_blocks(step))

        val_loss /= val_steps

        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)

        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.6f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)

step:0/5800 val_loss:10.825840 train_time:0ms step_avg:0.13ms

step:125/5800 val_loss:4.289520 train_time:28222ms step_avg:225.77ms

step:250/5800 val_loss:3.877428 train_time:56483ms step_avg:225.93ms

step:375/5800 val_loss:3.696738 train_time:85311ms step_avg:227.50ms

step:500/5800 val_loss:3.580342 train_time:114494ms step_avg:228.99ms

step:625/5800 val_loss:3.501325 train_time:143892ms step_avg:230.23ms

step:750/5800 val_loss:3.445790 train_time:173970ms step_avg:231.96ms

step:875/5800 val_loss:3.398829 train_time:203829ms step_avg:232.95ms

step:1000/5800 val_loss:3.359708 train_time:233824ms step_avg:233.82ms

step:1125/5800 val_loss:3.331579 train_time:263941ms step_avg:234.61ms

step:1250/5800 val_loss:3.305048 train_time:294038ms step_avg:235.23ms

step:1375/5800 val_loss:3.286346 train_time:324503ms step_avg:236.00ms

step:1500/5800 val_loss:3.263872 train_time:354913ms step_avg:236.61ms

step:1625/5800 val_loss:3.248456 train_time:385433ms step_avg:237.19ms

step:1750/5800 val_loss:3.232831 train_time:416202ms step_avg:237.83ms

step:1875/5800 val_loss:3.214183 train_time:447003ms step_avg:238.40ms

step:2000/5800 val_loss:3.198055 train_time:477631ms step_avg:238.82ms

step:2125/5800 val_loss:3.182905 train_time:508401ms step_avg:239.25ms

step:2250/5800 val_loss:3.168586 train_time:539150ms step_avg:239.62ms

step:2375/5800 val_loss:3.156276 train_time:570131ms step_avg:240.06ms

step:2500/5800 val_loss:3.145521 train_time:601139ms step_avg:240.46ms

step:2625/5800 val_loss:3.132375 train_time:631763ms step_avg:240.67ms

step:2750/5800 val_loss:3.121898 train_time:662381ms step_avg:240.87ms

step:2875/5800 val_loss:3.111240 train_time:693075ms step_avg:241.07ms

step:3000/5800 val_loss:3.100978 train_time:723762ms step_avg:241.25ms

step:3125/5800 val_loss:3.089165 train_time:754469ms step_avg:241.43ms

step:3250/5800 val_loss:3.078499 train_time:785131ms step_avg:241.58ms

step:3375/5800 val_loss:3.068614 train_time:816003ms step_avg:241.78ms

step:3500/5800 val_loss:3.060379 train_time:846720ms step_avg:241.92ms

step:3625/5800 val_loss:3.051567 train_time:877834ms step_avg:242.16ms

step:3750/5800 val_loss:3.043314 train_time:908607ms step_avg:242.30ms

step:3875/5800 val_loss:3.033554 train_time:939369ms step_avg:242.42ms

step:4000/5800 val_loss:3.024500 train_time:970437ms step_avg:242.61ms

step:4125/5800 val_loss:3.016577 train_time:1001222ms step_avg:242.72ms

step:4250/5800 val_loss:3.008387 train_time:1031935ms step_avg:242.81ms

step:4375/5800 val_loss:2.999895 train_time:1062840ms step_avg:242.93ms

step:4500/5800 val_loss:2.991480 train_time:1093824ms step_avg:243.07ms

step:4625/5800 val_loss:2.982606 train_time:1124840ms step_avg:243.21ms

step:4750/5800 val_loss:2.973720 train_time:1155852ms step_avg:243.34ms

step:4875/5800 val_loss:2.965306 train_time:1187247ms step_avg:243.54ms

step:5000/5800 val_loss:2.957200 train_time:1218586ms step_avg:243.72ms

step:5125/5800 val_loss:2.949435 train_time:1250122ms step_avg:243.93ms

step:5250/5800 val_loss:2.941967 train_time:1281826ms step_avg:244.16ms

step:5375/5800 val_loss:2.935326 train_time:1313594ms step_avg:244.39ms

step:5500/5800 val_loss:2.929443 train_time:1345562ms step_avg:244.65ms

step:5625/5800 val_loss:2.924125 train_time:1377762ms step_avg:244.94ms

step:5750/5800 val_loss:2.920357 train_time:1410256ms step_avg:245.26ms

step:5800/5800 val_loss:2.919787 train_time:1423217ms step_avg:245.38ms


## 26 1

    val_loss_every = 125 # every how many steps to evaluate val loss? 0 for only at the end

    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):

        val_loss = 0

                val_loss += model(inputs, targets, get_window_size_blocks(step))

        val_loss /= val_steps

        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)

        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.6f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)

step:0/5800 val_loss:10.825840 train_time:0ms step_avg:0.30ms

step:125/5800 val_loss:4.293403 train_time:28181ms step_avg:225.45ms

step:250/5800 val_loss:3.872679 train_time:56503ms step_avg:226.01ms

step:375/5800 val_loss:3.696451 train_time:85289ms step_avg:227.44ms

step:500/5800 val_loss:3.580016 train_time:114327ms step_avg:228.65ms

step:625/5800 val_loss:3.499635 train_time:143688ms step_avg:229.90ms

step:750/5800 val_loss:3.443692 train_time:173439ms step_avg:231.25ms

step:875/5800 val_loss:3.397855 train_time:203265ms step_avg:232.30ms

step:1000/5800 val_loss:3.359243 train_time:233261ms step_avg:233.26ms

step:1125/5800 val_loss:3.331818 train_time:263265ms step_avg:234.01ms

step:1250/5800 val_loss:3.304567 train_time:293326ms step_avg:234.66ms

step:1375/5800 val_loss:3.283537 train_time:323733ms step_avg:235.44ms

step:1500/5800 val_loss:3.263742 train_time:354119ms step_avg:236.08ms

step:1625/5800 val_loss:3.248207 train_time:384586ms step_avg:236.67ms

step:1750/5800 val_loss:3.232780 train_time:415017ms step_avg:237.15ms

step:1875/5800 val_loss:3.214603 train_time:445498ms step_avg:237.60ms

step:2000/5800 val_loss:3.197401 train_time:475975ms step_avg:237.99ms

step:2125/5800 val_loss:3.182088 train_time:506719ms step_avg:238.46ms

step:2250/5800 val_loss:3.167800 train_time:537412ms step_avg:238.85ms

step:2375/5800 val_loss:3.155632 train_time:568143ms step_avg:239.22ms

step:2500/5800 val_loss:3.144672 train_time:598843ms step_avg:239.54ms

step:2625/5800 val_loss:3.131773 train_time:629482ms step_avg:239.80ms

step:2750/5800 val_loss:3.121606 train_time:660099ms step_avg:240.04ms

step:2875/5800 val_loss:3.111138 train_time:690740ms step_avg:240.26ms

step:3000/5800 val_loss:3.101046 train_time:721464ms step_avg:240.49ms

step:3125/5800 val_loss:3.090022 train_time:752219ms step_avg:240.71ms

step:3250/5800 val_loss:3.078840 train_time:782919ms step_avg:240.90ms

step:3375/5800 val_loss:3.068560 train_time:813599ms step_avg:241.07ms

step:3500/5800 val_loss:3.060320 train_time:844375ms step_avg:241.25ms

step:3625/5800 val_loss:3.051999 train_time:875112ms step_avg:241.41ms

step:3750/5800 val_loss:3.043605 train_time:905855ms step_avg:241.56ms

step:3875/5800 val_loss:3.034209 train_time:936586ms step_avg:241.70ms

step:4000/5800 val_loss:3.024969 train_time:967374ms step_avg:241.84ms

step:4125/5800 val_loss:3.016793 train_time:998044ms step_avg:241.95ms

step:4250/5800 val_loss:3.008399 train_time:1028785ms step_avg:242.07ms

step:4375/5800 val_loss:3.000082 train_time:1059706ms step_avg:242.22ms

step:4500/5800 val_loss:2.991759 train_time:1090732ms step_avg:242.38ms

step:4625/5800 val_loss:2.982873 train_time:1121789ms step_avg:242.55ms

step:4750/5800 val_loss:2.974193 train_time:1152771ms step_avg:242.69ms

step:4875/5800 val_loss:2.965691 train_time:1184127ms step_avg:242.90ms

step:5000/5800 val_loss:2.957644 train_time:1215478ms step_avg:243.10ms

step:5125/5800 val_loss:2.950069 train_time:1247015ms step_avg:243.32ms

step:5250/5800 val_loss:2.942570 train_time:1278702ms step_avg:243.56ms

step:5375/5800 val_loss:2.935865 train_time:1310418ms step_avg:243.80ms

step:5500/5800 val_loss:2.929812 train_time:1342333ms step_avg:244.06ms

step:5625/5800 val_loss:2.924613 train_time:1374445ms step_avg:244.35ms

step:5750/5800 val_loss:2.920847 train_time:1406876ms step_avg:244.67ms

step:5800/5800 val_loss:2.920263 train_time:1419822ms step_avg:244.80ms


## 26 2

    val_loss_every = 125 # every how many steps to evaluate val loss? 0 for only at the end

    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):

        val_loss = 0

                val_loss += model(inputs, targets, get_window_size_blocks(step))

        val_loss /= val_steps

        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)

        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.6f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)

step:0/5800 val_loss:10.825840 train_time:0ms step_avg:0.13ms

step:125/5800 val_loss:4.296202 train_time:28175ms step_avg:225.40ms

step:250/5800 val_loss:3.882283 train_time:56343ms step_avg:225.37ms

step:375/5800 val_loss:3.697254 train_time:85315ms step_avg:227.51ms

step:500/5800 val_loss:3.579919 train_time:114456ms step_avg:228.91ms

step:625/5800 val_loss:3.499672 train_time:144017ms step_avg:230.43ms

step:750/5800 val_loss:3.442685 train_time:173823ms step_avg:231.76ms

step:875/5800 val_loss:3.397103 train_time:203607ms step_avg:232.69ms

step:1000/5800 val_loss:3.357398 train_time:233594ms step_avg:233.59ms

step:1125/5800 val_loss:3.330905 train_time:263654ms step_avg:234.36ms

step:1250/5800 val_loss:3.302062 train_time:293713ms step_avg:234.97ms

step:1375/5800 val_loss:3.281865 train_time:324122ms step_avg:235.73ms

step:1500/5800 val_loss:3.260487 train_time:354405ms step_avg:236.27ms

step:1625/5800 val_loss:3.244762 train_time:384888ms step_avg:236.85ms

step:1750/5800 val_loss:3.229276 train_time:415358ms step_avg:237.35ms

step:1875/5800 val_loss:3.211064 train_time:445835ms step_avg:237.78ms

step:2000/5800 val_loss:3.193983 train_time:476413ms step_avg:238.21ms

step:2125/5800 val_loss:3.179444 train_time:507156ms step_avg:238.66ms

step:2250/5800 val_loss:3.165385 train_time:537885ms step_avg:239.06ms

step:2375/5800 val_loss:3.153061 train_time:568608ms step_avg:239.41ms

step:2500/5800 val_loss:3.141919 train_time:599281ms step_avg:239.71ms

step:2625/5800 val_loss:3.129208 train_time:629970ms step_avg:239.99ms

step:2750/5800 val_loss:3.119112 train_time:660638ms step_avg:240.23ms

step:2875/5800 val_loss:3.108303 train_time:691376ms step_avg:240.48ms

step:3000/5800 val_loss:3.098024 train_time:722108ms step_avg:240.70ms

step:3125/5800 val_loss:3.086675 train_time:752872ms step_avg:240.92ms

step:3250/5800 val_loss:3.076471 train_time:783570ms step_avg:241.10ms

step:3375/5800 val_loss:3.067063 train_time:814276ms step_avg:241.27ms

step:3500/5800 val_loss:3.057935 train_time:845036ms step_avg:241.44ms

step:3625/5800 val_loss:3.049607 train_time:875807ms step_avg:241.60ms

step:3750/5800 val_loss:3.040952 train_time:906472ms step_avg:241.73ms

step:3875/5800 val_loss:3.032055 train_time:937165ms step_avg:241.85ms

step:4000/5800 val_loss:3.022520 train_time:967876ms step_avg:241.97ms

step:4125/5800 val_loss:3.014706 train_time:998534ms step_avg:242.07ms

step:4250/5800 val_loss:3.006573 train_time:1029228ms step_avg:242.17ms

step:4375/5800 val_loss:2.997779 train_time:1060117ms step_avg:242.31ms

step:4500/5800 val_loss:2.989719 train_time:1091146ms step_avg:242.48ms

step:4625/5800 val_loss:2.980757 train_time:1122214ms step_avg:242.64ms

step:4750/5800 val_loss:2.971955 train_time:1153260ms step_avg:242.79ms

step:4875/5800 val_loss:2.963423 train_time:1184579ms step_avg:242.99ms

step:5000/5800 val_loss:2.955475 train_time:1215963ms step_avg:243.19ms

step:5125/5800 val_loss:2.947706 train_time:1247480ms step_avg:243.41ms

step:5250/5800 val_loss:2.940139 train_time:1279176ms step_avg:243.65ms

step:5375/5800 val_loss:2.933371 train_time:1310924ms step_avg:243.89ms

step:5500/5800 val_loss:2.927552 train_time:1342857ms step_avg:244.16ms

step:5625/5800 val_loss:2.922315 train_time:1375050ms step_avg:244.45ms

step:5750/5800 val_loss:2.918552 train_time:1407518ms step_avg:244.79ms

step:5800/5800 val_loss:2.917983 train_time:1420470ms step_avg:244.91ms


## 26 3

    val_loss_every = 125 # every how many steps to evaluate val loss? 0 for only at the end

    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):

        val_loss = 0

                val_loss += model(inputs, targets, get_window_size_blocks(step))

        val_loss /= val_steps

        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)

        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.6f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)

step:0/5800 val_loss:10.825840 train_time:0ms step_avg:0.13ms

step:125/5800 val_loss:4.296066 train_time:28198ms step_avg:225.59ms

step:250/5800 val_loss:3.875204 train_time:56623ms step_avg:226.49ms

step:375/5800 val_loss:3.696008 train_time:85343ms step_avg:227.58ms

step:500/5800 val_loss:3.577362 train_time:114373ms step_avg:228.75ms

step:625/5800 val_loss:3.499145 train_time:143745ms step_avg:229.99ms

step:750/5800 val_loss:3.443357 train_time:173495ms step_avg:231.33ms

step:875/5800 val_loss:3.397771 train_time:203305ms step_avg:232.35ms

step:1000/5800 val_loss:3.356456 train_time:233443ms step_avg:233.44ms

step:1125/5800 val_loss:3.329739 train_time:263452ms step_avg:234.18ms

step:1250/5800 val_loss:3.303415 train_time:293511ms step_avg:234.81ms

step:1375/5800 val_loss:3.282091 train_time:323912ms step_avg:235.57ms

step:1500/5800 val_loss:3.259538 train_time:354286ms step_avg:236.19ms

step:1625/5800 val_loss:3.245869 train_time:384748ms step_avg:236.77ms

step:1750/5800 val_loss:3.228784 train_time:415196ms step_avg:237.25ms

step:1875/5800 val_loss:3.212082 train_time:445702ms step_avg:237.71ms

step:2000/5800 val_loss:3.195460 train_time:476175ms step_avg:238.09ms

step:2125/5800 val_loss:3.180680 train_time:506822ms step_avg:238.50ms

step:2250/5800 val_loss:3.166107 train_time:537461ms step_avg:238.87ms

step:2375/5800 val_loss:3.153818 train_time:568081ms step_avg:239.19ms

step:2500/5800 val_loss:3.142874 train_time:598714ms step_avg:239.49ms

step:2625/5800 val_loss:3.129431 train_time:629332ms step_avg:239.75ms

step:2750/5800 val_loss:3.119046 train_time:659951ms step_avg:239.98ms

step:2875/5800 val_loss:3.108873 train_time:690590ms step_avg:240.21ms

step:3000/5800 val_loss:3.099116 train_time:721309ms step_avg:240.44ms

step:3125/5800 val_loss:3.087420 train_time:752076ms step_avg:240.66ms

step:3250/5800 val_loss:3.076606 train_time:782733ms step_avg:240.84ms

step:3375/5800 val_loss:3.066951 train_time:813420ms step_avg:241.01ms

step:3500/5800 val_loss:3.058094 train_time:844108ms step_avg:241.17ms

step:3625/5800 val_loss:3.050452 train_time:874785ms step_avg:241.32ms

step:3750/5800 val_loss:3.041585 train_time:905449ms step_avg:241.45ms

step:3875/5800 val_loss:3.032032 train_time:936170ms step_avg:241.59ms

step:4000/5800 val_loss:3.022784 train_time:966937ms step_avg:241.73ms

step:4125/5800 val_loss:3.014500 train_time:997674ms step_avg:241.86ms

step:4250/5800 val_loss:3.006442 train_time:1028442ms step_avg:241.99ms

step:4375/5800 val_loss:2.998192 train_time:1059377ms step_avg:242.14ms

step:4500/5800 val_loss:2.989624 train_time:1090334ms step_avg:242.30ms

step:4625/5800 val_loss:2.980731 train_time:1121330ms step_avg:242.45ms

step:4750/5800 val_loss:2.972144 train_time:1152385ms step_avg:242.61ms

step:4875/5800 val_loss:2.963665 train_time:1183759ms step_avg:242.82ms

step:5000/5800 val_loss:2.955751 train_time:1215167ms step_avg:243.03ms

step:5125/5800 val_loss:2.947949 train_time:1246703ms step_avg:243.26ms

step:5250/5800 val_loss:2.940351 train_time:1278331ms step_avg:243.49ms

step:5375/5800 val_loss:2.933583 train_time:1310016ms step_avg:243.72ms

step:5500/5800 val_loss:2.927682 train_time:1341966ms step_avg:243.99ms

step:5625/5800 val_loss:2.922412 train_time:1374134ms step_avg:244.29ms

step:5750/5800 val_loss:2.918637 train_time:1406558ms step_avg:244.62ms

step:5800/5800 val_loss:2.918085 train_time:1419500ms step_avg:244.74ms


## 26 4

    val_loss_every = 125 # every how many steps to evaluate val loss? 0 for only at the end

    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):

        val_loss = 0

                val_loss += model(inputs, targets, get_window_size_blocks(step))

        val_loss /= val_steps

        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)

        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.6f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)

step:0/5800 val_loss:10.825840 train_time:0ms step_avg:0.15ms

step:125/5800 val_loss:4.302063 train_time:28194ms step_avg:225.55ms

step:250/5800 val_loss:3.876690 train_time:56632ms step_avg:226.53ms

step:375/5800 val_loss:3.698671 train_time:85348ms step_avg:227.59ms

step:500/5800 val_loss:3.580584 train_time:114513ms step_avg:229.03ms

step:625/5800 val_loss:3.502209 train_time:143897ms step_avg:230.24ms

step:750/5800 val_loss:3.446772 train_time:173777ms step_avg:231.70ms

step:875/5800 val_loss:3.399667 train_time:203631ms step_avg:232.72ms

step:1000/5800 val_loss:3.360105 train_time:233632ms step_avg:233.63ms

step:1125/5800 val_loss:3.331693 train_time:263790ms step_avg:234.48ms

step:1250/5800 val_loss:3.304243 train_time:293830ms step_avg:235.06ms

step:1375/5800 val_loss:3.283419 train_time:324166ms step_avg:235.76ms

step:1500/5800 val_loss:3.263976 train_time:354449ms step_avg:236.30ms

step:1625/5800 val_loss:3.247225 train_time:384853ms step_avg:236.83ms

step:1750/5800 val_loss:3.230798 train_time:415310ms step_avg:237.32ms

step:1875/5800 val_loss:3.213994 train_time:445814ms step_avg:237.77ms

step:2000/5800 val_loss:3.197791 train_time:476383ms step_avg:238.19ms

step:2125/5800 val_loss:3.181765 train_time:507118ms step_avg:238.64ms

step:2250/5800 val_loss:3.167682 train_time:537829ms step_avg:239.03ms

step:2375/5800 val_loss:3.155754 train_time:568540ms step_avg:239.39ms

step:2500/5800 val_loss:3.144059 train_time:599168ms step_avg:239.67ms

step:2625/5800 val_loss:3.131675 train_time:629774ms step_avg:239.91ms

step:2750/5800 val_loss:3.121688 train_time:660383ms step_avg:240.14ms

step:2875/5800 val_loss:3.110390 train_time:691083ms step_avg:240.38ms

step:3000/5800 val_loss:3.100665 train_time:721731ms step_avg:240.58ms

step:3125/5800 val_loss:3.089355 train_time:752436ms step_avg:240.78ms

step:3250/5800 val_loss:3.078197 train_time:783157ms step_avg:240.97ms

step:3375/5800 val_loss:3.068622 train_time:813837ms step_avg:241.14ms

step:3500/5800 val_loss:3.059895 train_time:844611ms step_avg:241.32ms

step:3625/5800 val_loss:3.051696 train_time:875358ms step_avg:241.48ms

step:3750/5800 val_loss:3.042981 train_time:906110ms step_avg:241.63ms

step:3875/5800 val_loss:3.033641 train_time:936867ms step_avg:241.77ms

step:4000/5800 val_loss:3.024306 train_time:967651ms step_avg:241.91ms

step:4125/5800 val_loss:3.016263 train_time:998336ms step_avg:242.02ms

step:4250/5800 val_loss:3.008216 train_time:1029001ms step_avg:242.12ms

step:4375/5800 val_loss:2.999825 train_time:1059968ms step_avg:242.28ms

step:4500/5800 val_loss:2.991349 train_time:1090913ms step_avg:242.43ms

step:4625/5800 val_loss:2.982351 train_time:1121981ms step_avg:242.59ms

step:4750/5800 val_loss:2.973421 train_time:1153038ms step_avg:242.74ms

step:4875/5800 val_loss:2.965122 train_time:1184360ms step_avg:242.95ms

step:5000/5800 val_loss:2.957097 train_time:1215669ms step_avg:243.13ms

step:5125/5800 val_loss:2.949403 train_time:1247181ms step_avg:243.35ms

step:5250/5800 val_loss:2.941895 train_time:1278861ms step_avg:243.59ms

step:5375/5800 val_loss:2.935165 train_time:1310577ms step_avg:243.83ms

step:5500/5800 val_loss:2.929192 train_time:1342510ms step_avg:244.09ms

step:5625/5800 val_loss:2.923907 train_time:1374696ms step_avg:244.39ms

step:5750/5800 val_loss:2.920210 train_time:1407187ms step_avg:244.73ms

step:5800/5800 val_loss:2.919636 train_time:1420127ms step_avg:244.85ms


## 26 5

    val_loss_every = 125 # every how many steps to evaluate val loss? 0 for only at the end

    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):

        val_loss = 0

                val_loss += model(inputs, targets, get_window_size_blocks(step))

        val_loss /= val_steps

        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)

        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.6f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)

step:0/5800 val_loss:10.825840 train_time:0ms step_avg:0.27ms

step:125/5800 val_loss:4.298494 train_time:28166ms step_avg:225.33ms

step:250/5800 val_loss:3.878978 train_time:56450ms step_avg:225.80ms

step:375/5800 val_loss:3.701263 train_time:85223ms step_avg:227.26ms

step:500/5800 val_loss:3.583069 train_time:114347ms step_avg:228.69ms

step:625/5800 val_loss:3.504774 train_time:143815ms step_avg:230.10ms

step:750/5800 val_loss:3.450564 train_time:173562ms step_avg:231.42ms

step:875/5800 val_loss:3.401689 train_time:203474ms step_avg:232.54ms

step:1000/5800 val_loss:3.362407 train_time:233587ms step_avg:233.59ms

step:1125/5800 val_loss:3.334236 train_time:263845ms step_avg:234.53ms

step:1250/5800 val_loss:3.306665 train_time:293907ms step_avg:235.13ms

step:1375/5800 val_loss:3.286640 train_time:324331ms step_avg:235.88ms

step:1500/5800 val_loss:3.266370 train_time:354608ms step_avg:236.41ms

step:1625/5800 val_loss:3.250893 train_time:385133ms step_avg:237.01ms

step:1750/5800 val_loss:3.233242 train_time:415713ms step_avg:237.55ms

step:1875/5800 val_loss:3.215568 train_time:446184ms step_avg:237.96ms

step:2000/5800 val_loss:3.199866 train_time:476735ms step_avg:238.37ms

step:2125/5800 val_loss:3.184150 train_time:507398ms step_avg:238.78ms

step:2250/5800 val_loss:3.169971 train_time:538002ms step_avg:239.11ms

step:2375/5800 val_loss:3.157521 train_time:568696ms step_avg:239.45ms

step:2500/5800 val_loss:3.146432 train_time:599320ms step_avg:239.73ms

step:2625/5800 val_loss:3.133031 train_time:629924ms step_avg:239.97ms

step:2750/5800 val_loss:3.122372 train_time:660501ms step_avg:240.18ms

step:2875/5800 val_loss:3.112367 train_time:691211ms step_avg:240.42ms

step:3000/5800 val_loss:3.101508 train_time:721913ms step_avg:240.64ms

step:3125/5800 val_loss:3.090298 train_time:752660ms step_avg:240.85ms

step:3250/5800 val_loss:3.079685 train_time:783340ms step_avg:241.03ms

step:3375/5800 val_loss:3.069775 train_time:814010ms step_avg:241.19ms

step:3500/5800 val_loss:3.060697 train_time:844757ms step_avg:241.36ms

step:3625/5800 val_loss:3.052545 train_time:875487ms step_avg:241.51ms

step:3750/5800 val_loss:3.043977 train_time:906217ms step_avg:241.66ms

step:3875/5800 val_loss:3.033692 train_time:936958ms step_avg:241.80ms

step:4000/5800 val_loss:3.024971 train_time:967733ms step_avg:241.93ms

step:4125/5800 val_loss:3.016851 train_time:998481ms step_avg:242.06ms

step:4250/5800 val_loss:3.008432 train_time:1029223ms step_avg:242.17ms

step:4375/5800 val_loss:2.999946 train_time:1060080ms step_avg:242.30ms

step:4500/5800 val_loss:2.991747 train_time:1091049ms step_avg:242.46ms

step:4625/5800 val_loss:2.982670 train_time:1122028ms step_avg:242.60ms

step:4750/5800 val_loss:2.973899 train_time:1153056ms step_avg:242.75ms

step:4875/5800 val_loss:2.965439 train_time:1184409ms step_avg:242.96ms

step:5000/5800 val_loss:2.957440 train_time:1215703ms step_avg:243.14ms

step:5125/5800 val_loss:2.949724 train_time:1247132ms step_avg:243.34ms

step:5250/5800 val_loss:2.942219 train_time:1278826ms step_avg:243.59ms

step:5375/5800 val_loss:2.935521 train_time:1310501ms step_avg:243.81ms

step:5500/5800 val_loss:2.929636 train_time:1342370ms step_avg:244.07ms

step:5625/5800 val_loss:2.924269 train_time:1374480ms step_avg:244.35ms

step:5750/5800 val_loss:2.920548 train_time:1406914ms step_avg:244.68ms

step:5800/5800 val_loss:2.919984 train_time:1419860ms step_avg:244.80ms


## 26 6

    val_loss_every = 125 # every how many steps to evaluate val loss? 0 for only at the end

    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):

        val_loss = 0

                val_loss += model(inputs, targets, get_window_size_blocks(step))

        val_loss /= val_steps

        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)

        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.6f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)

step:0/5800 val_loss:10.825840 train_time:0ms step_avg:0.15ms

step:125/5800 val_loss:4.289963 train_time:28186ms step_avg:225.49ms

step:250/5800 val_loss:3.874116 train_time:56574ms step_avg:226.30ms

step:375/5800 val_loss:3.697003 train_time:85228ms step_avg:227.27ms

step:500/5800 val_loss:3.578802 train_time:114522ms step_avg:229.04ms

step:625/5800 val_loss:3.501460 train_time:143877ms step_avg:230.20ms

step:750/5800 val_loss:3.443288 train_time:173665ms step_avg:231.55ms

step:875/5800 val_loss:3.398944 train_time:203492ms step_avg:232.56ms

step:1000/5800 val_loss:3.358758 train_time:233506ms step_avg:233.51ms

step:1125/5800 val_loss:3.332333 train_time:263779ms step_avg:234.47ms

step:1250/5800 val_loss:3.304260 train_time:293803ms step_avg:235.04ms

step:1375/5800 val_loss:3.284369 train_time:324114ms step_avg:235.72ms

step:1500/5800 val_loss:3.262934 train_time:354478ms step_avg:236.32ms

step:1625/5800 val_loss:3.248307 train_time:384971ms step_avg:236.91ms

step:1750/5800 val_loss:3.231253 train_time:415419ms step_avg:237.38ms

step:1875/5800 val_loss:3.213700 train_time:445931ms step_avg:237.83ms

step:2000/5800 val_loss:3.197685 train_time:476495ms step_avg:238.25ms

step:2125/5800 val_loss:3.181862 train_time:507186ms step_avg:238.68ms

step:2250/5800 val_loss:3.167390 train_time:537881ms step_avg:239.06ms

step:2375/5800 val_loss:3.155761 train_time:568611ms step_avg:239.42ms

step:2500/5800 val_loss:3.144498 train_time:599277ms step_avg:239.71ms

step:2625/5800 val_loss:3.131982 train_time:629974ms step_avg:239.99ms

step:2750/5800 val_loss:3.121144 train_time:660654ms step_avg:240.24ms

step:2875/5800 val_loss:3.110864 train_time:691396ms step_avg:240.49ms

step:3000/5800 val_loss:3.100978 train_time:722124ms step_avg:240.71ms

step:3125/5800 val_loss:3.089506 train_time:752822ms step_avg:240.90ms

step:3250/5800 val_loss:3.079010 train_time:783527ms step_avg:241.09ms

step:3375/5800 val_loss:3.069015 train_time:814228ms step_avg:241.25ms

step:3500/5800 val_loss:3.060230 train_time:844987ms step_avg:241.42ms

step:3625/5800 val_loss:3.051967 train_time:875769ms step_avg:241.59ms

step:3750/5800 val_loss:3.042726 train_time:906440ms step_avg:241.72ms

step:3875/5800 val_loss:3.033427 train_time:937109ms step_avg:241.83ms

step:4000/5800 val_loss:3.024497 train_time:967804ms step_avg:241.95ms

step:4125/5800 val_loss:3.016171 train_time:998468ms step_avg:242.05ms

step:4250/5800 val_loss:3.007999 train_time:1029216ms step_avg:242.17ms

step:4375/5800 val_loss:2.999842 train_time:1060093ms step_avg:242.31ms

step:4500/5800 val_loss:2.991397 train_time:1091057ms step_avg:242.46ms

step:4625/5800 val_loss:2.982648 train_time:1122049ms step_avg:242.61ms

step:4750/5800 val_loss:2.973885 train_time:1153104ms step_avg:242.76ms

step:4875/5800 val_loss:2.965296 train_time:1184507ms step_avg:242.98ms

step:5000/5800 val_loss:2.957347 train_time:1215915ms step_avg:243.18ms

step:5125/5800 val_loss:2.949855 train_time:1247428ms step_avg:243.40ms

step:5250/5800 val_loss:2.942336 train_time:1279137ms step_avg:243.65ms

step:5375/5800 val_loss:2.935542 train_time:1310880ms step_avg:243.88ms

step:5500/5800 val_loss:2.929498 train_time:1342803ms step_avg:244.15ms

step:5625/5800 val_loss:2.924226 train_time:1374924ms step_avg:244.43ms

step:5750/5800 val_loss:2.920458 train_time:1407371ms step_avg:244.76ms

step:5800/5800 val_loss:2.919901 train_time:1420331ms step_avg:244.88ms