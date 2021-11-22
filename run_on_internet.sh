CUDA_VISIBLE_DEVICES=0 python dynaboa_internet.py --expdir exps --expname internet --dataset internet \
                                            --motionloss_weight 0.8 \
                                            --retrieval 1 \
                                            --dynamic_boa 1 \
                                            --optim_steps 7 \
                                            --cos_sim_threshold 3.1e-4 \
                                            --shape_prior_weight 2e-4 \
                                            --pose_prior_weight 1e-4 \
                                            --save_res 1
