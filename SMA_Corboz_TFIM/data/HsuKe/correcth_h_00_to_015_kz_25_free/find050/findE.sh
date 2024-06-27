# python findminE.py --prefix kitaev_datas/kz_4_h_00_anti/kitaev-D4-z4-h00- --min 1 --max 50
# python findminE.py --prefix kitaev_datas/kz_4_h_005_anti/kitaev-D4-z4-h005- --min 1 --max 50
# python findminE.py --prefix kitaev_datas/kz_4_h_01_anti/kitaev-D4-z4-h01- --min 1 --max 50
# python findminE.py --prefix kitaev_datas/kz_4_h_015_anti/kitaev-D4-z4-h015- --min 1 --max 50
# python findminE.py --prefix kitaev_datas/kz_4_h_02_anti/kitaev-D4-z4-h02- --min 1 --max 50
# python findminE.py --prefix kitaev_datas/kz_4_h_025_anti/kitaev-D4-z4-h025- --min 1 --max 50
# python findminE.py --prefix kitaev_datas/kz_4_h_03_anti/kitaev-D4-z4-h03- --min 1 --max 50
# python findminE.py --prefix kitaev_datas/kz_4_h_035_anti/kitaev-D4-z4-h035- --min 1 --max 50
# python findminE.py --prefix kitaev_datas/kz_4_h_04_anti/kitaev-D4-z4-h04- --min 1 --max 50
# python findminE.py --prefix kitaev_datas/kz_4_h_045_anti/kitaev-D4-z4-h045- --min 1 --max 50
# python findminE.py --prefix kitaev_datas/kz_4_h_05_anti/kitaev-D4-z4-h05- --min 1 --max 50
# python findminE.py --prefix kitaev_datas/kz_4_h_055_anti/kitaev-D4-z4-h055- --min 1 --max 50
# python findminE.py --prefix kitaev_datas/kz_4_h_06_anti/kitaev-D4-z4-h06- --min 1 --max 50

#python findminE.py --prefix kitaev_datas/test_kz_25_h_0_ferro/kitaev-D4-z25-h0- --min 1 --max 50
# python findminE.py --prefix kitaev_datas/test_kx_25_h_0_ferro/kitaev-D4-z25-h0- --min 1 --max 11
#python findminE.py --prefix kitaev_datas/test_kz_1_h_0_anti/kitaev-D4-z1-h0- --min 1 --max 7
prefix=h_050-
python ../../../findminE.py --prefix ${prefix} --min 1 --max 27

# for (( i=3; i<=14; i++ ))
#     do
#         rm ${prefix}${i}_checkpoint.p
#         rm ${prefix}${i}_state.json
#         rm ${prefix}${i}.log
#     done
# z4
# list=(42 1 30 38 27 46 46)
# h=(3 35 4 45 5 55 6)

# z2.5

# list=(35 35 19 48 21 39 27 43 43 16 16 47 47)
# h=(0 05 1 15 2 25 3 35 4 45 5 55 6)

# for k in {0..12}
#     do
#         for (( i=1; i<=${list[k]}; i++ ))
#             do
#                 rm kitaev_datas/kz_25_h_0${h[k]}_anti/kitaev-D4-z25-h0${h[k]}-${i}_checkpoint.p
#                 rm kitaev_datas/kz_25_h_0${h[k]}_anti/kitaev-D4-z25-h0${h[k]}-${i}_state.json
#                 rm kitaev_datas/kz_25_h_0${h[k]}_anti/kitaev-D4-z25-h0${h[k]}-${i}.log
#             done
#         for (( i=${list[k]}+2; i<=50; i++ ))
#             do
#                 rm kitaev_datas/kz_25_h_0${h[k]}_anti/kitaev-D4-z25-h0${h[k]}-${i}_checkpoint.p
#                 rm kitaev_datas/kz_25_h_0${h[k]}_anti/kitaev-D4-z25-h0${h[k]}-${i}_state.json
#                 rm kitaev_datas/kz_25_h_0${h[k]}_anti/kitaev-D4-z25-h0${h[k]}-${i}.log
#             done
#     done

# for (( i=1; i<=47; i++ ))
#     do
#         rm kitaev_datas/test_kz_25_h_0_ferro/kitaev-D4-z25-h0-${i}_checkpoint.p
#         rm kitaev_datas/test_kz_25_h_0_ferro/kitaev-D4-z25-h0-${i}_state.json
#         rm kitaev_datas/test_kz_25_h_0_ferro/kitaev-D4-z25-h0-${i}.log
#     done

# for (( i=49; i<=50; i++ ))
#     do
#         rm kitaev_datas/test_kz_25_h_0_ferro/kitaev-D4-z25-h0-${i}_checkpoint.p
#         rm kitaev_datas/test_kz_25_h_0_ferro/kitaev-D4-z25-h0-${i}_state.json
#         rm kitaev_datas/test_kz_25_h_0_ferro/kitaev-D4-z25-h0-${i}.log
#     done