# k-gamma-gtno
此專案融合了 https://github.com/weilintu/peps-excitation 與 https://github.com/j9263178/k-gamma-gtno
最主要的檔案在 `SMA_Corboz_TFIM` 資料夾中

詳細解釋的論文：http://tdr.lib.ntu.edu.tw/jspui/handle/123456789/93994

## Ground State
此專案必須使用已經計算好的 ground state 的 PEPS 才能進行
目前所使用的 ground state 在 https://github.com/j9263178/ipeps/tree/master/ad/kitaev_datas 中(Kitaev model)

## Excited State (此專案主要使用 https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.5.010335 的方法)
此方法是Wei-Lin Tu的方法，與Corboz group的方法關係不大，只是資料夾命名問題

所有模擬資料會在 `SMA_Corboz_TFIM/data` 中
你唯一需要使用的檔案就是 `sweep_SWandexcitedE_TFIM_withP_onceforall.sh` 或者 `sweep_SWandexcitedE_TFIM_Pderv_onceforall.sh`，最終再使用`runAll.sh`來執行所有參數的模擬
`withP` 是指使用原始Wei-Lin Tu的方法，做微分的時候並沒有微分到CTMRG的Projector
`Pderv` 是指參考使用Corboz的[改進方法](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.108.195111)，做微分的時候有微分到CTMRG的Projector

### 使用方法
使用conda或mamba建立含有pytorch的環境，並且安裝任何提示缺失的package(不包括ArrayFire，用不到)

### 執行檔案
```bash
bash sweep_SWandexcitedE_TFIM_withP_onceforall.sh
```
或者
```bash
bash sweep_SWandexcitedE_TFIM_Pderv_onceforall.sh
```

### 參數設定
在 `sweep_SWandexcitedE_TFIM_withP_onceforall.sh` 中
`SMAMethod`和`StoredMatMethod`分別指定了要用哪個python檔案來計算SMA(Single Mode Approximation 並且儲存 Hamiltonian Matrix 和 Norm Matrix)
以及 如何利用儲存的矩陣來計算Excitation Spectrum，Static Structure Factor和Dynamical Spectral Function等等。
並且請記得，您必須傳入參數指定各項參數，具體使用方法建議參考`runAll.sh`

`sweep_SWandexcitedE_TFIM_Pderv_onceforall.sh`結構與`sweep_SWandexcitedE_TFIM_withP_onceforall.sh`結構相同

### runDraw 參數
`runDraw` 是指是否要畫圖，若要畫圖，請設定為 `True`，若不畫圖，請設定為 `False`

請特別關注在bash檔案中具體實現runDraw的程式碼，若要畫出想要的圖，請修改會傳入graph.py的x-tick參數(例如`"[0,2,3,4,6,7], [r'\$M(\pi,0)\$', r'\$X(\pi,\pi)\$',r'\$S(\frac{\pi}{2},\frac{\pi}{2})\$', r'\$\Gamma(0,0)\$', r'\$M(\pi,0)\$', r'\$S(\frac{\pi}{2},\frac{\pi}{2})\$']"`即代表x軸的刻度與label)

具體畫圖實現請參考`SMA_Corboz_TFIM/graph.py`
畫完圖後會將圖存入在`sweep_SWandexcitedE_TFIM_withP_onceforall.sh`中 傳入graph.py的`figurepath`參數所指定的資料夾中

### runGUPTRI 參數
`runGUPTRI` 是指是否要使用GUPTRI來計算Excitation Spectrum
此參數基本上已經不使用，因為我們發現此方法並無法得到更好的結果

### statefile 參數
`statefile` 是指 ground state 的 PEPS 儲存檔案名稱，是在`datadir`資料夾中的檔案名稱