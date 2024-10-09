#!/bin/bash
# WithP
# Kz=1.0
# for h in 0.00 0.05 0.10 0.15
# do
#     for eig_size in 1 12 26 50 75 100 125 500
#     do
#         bash sweep_SWandexcitedE_KitaevAnyon_free_withP_onceforall.sh ${Kz} ${h} data/HsuKe/correcth_h_00_to_015_kz_1_free cpu True True True ${eig_size} False
#     done
# done
# Kz=2.5
# for h in 0.00 0.08 0.20 0.40
# do
#     for eig_size in 1 2 3 4 5 6 12 26 50 75 100 125 500
#     do
#         bash sweep_SWandexcitedE_KitaevAnyon_free_withP_onceforall.sh ${Kz} ${h} data/HsuKe/correcth_h_00_to_015_kz_25_free cpu True True True ${eig_size} False
#     done
# done

# Pderv
# Kz=1.0
# for h in 0.00 0.05 0.10 0.15
# do
#     for eig_size in 1 50 75 125
#     # for eig_size in 1
#     do
#         bash sweep_SWandexcitedE_KitaevAnyon_free_Pderv_onceforall.sh ${Kz} ${h} data/HsuKe/correcth_h_00_to_015_kz_1_free cpu True True True ${eig_size} False
#     done
# done


# Kz=1.0
# for h in 0.00 0.05 0.10 0.15
# do
#     for eig_size in 1 50 75 125
#     do
#         bash sweep_SWandexcitedE_KitaevAnyon_free_withP_onceforall.sh ${Kz} ${h} data/HsuKe/correcth_h_00_to_015_kz_1_free cpu True True True ${eig_size} False
#     done
# done


# Kz=2.5
# for h in 0.00 0.08 0.20 0.40
# do
#     for eig_size in 1
#     do
#         bash sweep_SWandexcitedE_KitaevAnyon_free_withP_onceforall.sh ${Kz} ${h} data/HsuKe/correcth_h_00_to_015_kz_25_free cpu True True True ${eig_size} False
#     done
# done

# hx=2.5
# chi=8
# for eig_size in 1 3 5
# do
#     bash sweep_SWandexcitedE_TFIM_withP_onceforall.sh ${hx} 0.0 data cpu True True True ${eig_size} ${chi}
# done

# hx=3.0
# chi=16
# for eig_size in 1 3 10
# do
#     bash sweep_SWandexcitedE_TFIM_withP_onceforall.sh ${hx} 0.0 data cpu True True True ${eig_size} ${chi}
# done

# hx=2.5
# chi=8
# for eig_size in 1 3 5
# do
#     bash sweep_SWandexcitedE_TFIM_Pderv_onceforall.sh ${hx} 0.0 data cpu True True True ${eig_size} ${chi}
# done
# hx=3.0
# chi=16
# for eig_size in 1 3 10
# do
#     bash sweep_SWandexcitedE_TFIM_Pderv_onceforall.sh ${hx} 0.0 data cpu True True True ${eig_size} ${chi}
# done

hx=2.5
chi=8
for eig_size in 500
do
    bash sweep_SWandexcitedE_TFIM_withP_onceforall.sh ${hx} 0.0 data cpu True True True ${eig_size} ${chi}
done
# hx=3.0
# chi=16
# for eig_size in 1 3 10
# do
#     bash sweep_SWandexcitedE_TFIM_withP_onceforall.sh ${hx} 0.0 data cpu True True True ${eig_size} ${chi}
# done


# Kz=1.0
# for h in 0.00 0.05 0.10
# do
#     for eig_size in 50
#     do
#         bash sweep_SWandexcitedE_KitaevAnyon_free_withP_onceforall.sh ${Kz} ${h} data/HsuKe/correcth_h_00_to_015_kz_1_free cpu True True True ${eig_size} False
#     done
# done
# Kz=1.0
# for h in 0.10
# do
#     for eig_size in 50
#     do
#         bash sweep_SWandexcitedE_KitaevAnyon_free_withP_onceforall.sh ${Kz} ${h} data/HsuKe/correcth_h_00_to_015_kz_1_free cpu True True True ${eig_size} False
#     done
# done

# Kz=2.5
# for h in 0.00 0.08 0.20 0.40
# do
#     for eig_size in 1
#     do
#         bash sweep_SWandexcitedE_KitaevAnyon_free_withP_onceforall.sh ${Kz} ${h} data/HsuKe/correcth_h_00_to_015_kz_25_free cpu True True True ${eig_size} False
#     done
# done
# Kz=2.5
# for h in 0.00 0.08
# do
#     for eig_size in 3
#     do
#         bash sweep_SWandexcitedE_KitaevAnyon_free_withP_onceforall.sh ${Kz} ${h} data/HsuKe/correcth_h_00_to_015_kz_25_free cpu True True True ${eig_size} False
#     done
# done
# Kz=2.5
# for h in 0.40
# do
#     for eig_size in 3
#     do
#         bash sweep_SWandexcitedE_KitaevAnyon_free_withP_onceforall.sh ${Kz} ${h} data/HsuKe/correcth_h_00_to_015_kz_25_free cpu True True True ${eig_size} False
#     done
# done

# Kz=1.0
# for h in 0.00 0.05 0.10
# do
#     for eig_size in 75
#     do
#         bash sweep_SWandexcitedE_KitaevAnyon_free_Pderv_onceforall.sh ${Kz} ${h} data/HsuKe/correcth_h_00_to_015_kz_1_free cpu True True True ${eig_size} False
#     done
# done
# Kz=1.0
# for h in 0.15
# do
#     for eig_size in 125
#     do
#         bash sweep_SWandexcitedE_KitaevAnyon_free_Pderv_onceforall.sh ${Kz} ${h} data/HsuKe/correcth_h_00_to_015_kz_1_free cpu True True True ${eig_size} False
#     done
# done