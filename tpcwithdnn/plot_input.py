# pylint: disable=too-many-statements
import os

from ROOT import TFile, TCanvas # pylint: disable=import-error, no-name-in-module
from ROOT import kFullSquare, kDot # pylint: disable=import-error, no-name-in-module

from tpcwithdnn.plot_utils import setup_frame

def draw_input(dirplots, draw_idc):
    if not os.path.isdir(dirplots):
        os.makedirs(dirplots)

    if draw_idc:
        dir_infix = "idc-study-202103/trees"
    else:
        dir_infix = "old-input-trees"
    f = TFile.Open("/mnt/temp/mkabus/%s/" % dir_infix +\
                   "treeInput_mean1.00_phi180_r65_z65.root","READ")
    t = f.Get("validation")

    t.SetMarkerStyle(kFullSquare)

    canvas = TCanvas()

    t.Draw("r:z:meanSC", "phi>0 && phi<3.14/9", "colz")
    setup_frame(None, "#it{z} (cm)", "#it{r} (cm)", "#it{<#rho>}_{SC} (fC/cm^{3})",
                x_offset=1.0, y_offset=1.2, label_size=0.035)
    canvas.SetMargin(0.1, 0.15, 0.1, 0.03) # left, right, bottom, top
    canvas.SaveAs("%s/r_z_meanSC_colz_phi_sector0.png" % dirplots)

    t.Draw("meanSC-flucSC:r:z>>htemp(65, 0, 250, 65, 83, 255)", "eventId == 0", "profcolz")
    setup_frame(None, "#it{z} (cm)", "#it{r} (cm)", "#it{#rho}_{SC} (fC/cm^{3})",
                x_offset=1.0, y_offset=1.2, label_size=0.035)
    canvas.SetMargin(0.1, 0.15, 0.1, 0.03) # left, right, bottom, top
    canvas.SaveAs("%s/r_z_randomSC_profcolz_phi_sector0.png" % dirplots)

    t.Draw("meanSC:r:phi>>htemp(180, 0., 6.28, 65, 83, 255)", "z>0 && z<1", "profcolz")
    setup_frame(None, "#it{#varphi} (rad)", "#it{r} (cm)", "#it{<#rho>}_{SC} (fC/cm^{3})",
                x_offset=1.0, y_offset=1.2, label_size=0.035)
    canvas.SetMargin(0.1, 0.15, 0.1, 0.03) # left, right, bottom, top
    canvas.SaveAs("%s/meanSC_r_phi_profcolz_z_0-1.png" % dirplots)

    t.Draw("meanSC:phi:r", "z>0 && z<1", "colz")
    setup_frame(None, "#it{#varphi} (rad)", "#it{<#rho>}_{SC} (fC/cm^{3})", "#it{r} (cm)",
                x_offset=1.0, y_offset=1.2, label_size=0.035)
    canvas.SetMargin(0.1, 0.15, 0.1, 0.03) # left, right, bottom, top
    canvas.SaveAs("%s/meanSC_phi_r_colz_z_0-1.png" % dirplots)

    t.Draw("r:z:meanDistR", "phi>0 && phi<3.14/9", "colz")
    setup_frame(None, "#it{z} (cm)", "#it{r} (cm)", "mean distortion <d#it{r}> (cm)",
                x_offset=1.0, y_offset=1.2, z_offset=1.2, label_size=0.035)
    canvas.SetMargin(0.1, 0.15, 0.1, 0.03) # left, right, bottom, top
    canvas.SaveAs("%s/r_z_meanDistR_colz_phi_sector0.png" % dirplots)

    t.Draw("r:z:meanDistRPhi", "phi>0 && phi<3.14/9", "colz")
    setup_frame(None, "#it{z} (cm)", "#it{r} (cm)", "mean distortion <d#it{r#varphi}> (cm)",
                x_offset=1.0, y_offset=1.2, z_offset=1.2, label_size=0.035)
    canvas.SetMargin(0.1, 0.15, 0.1, 0.03) # left, right, bottom, top
    canvas.SaveAs("%s/r_z_meanDistRPhi_colz_phi_sector0.png" % dirplots)

    t.Draw("r:z:meanDistZ", "phi>0 && phi<3.14/9", "colz")
    setup_frame(None, "#it{z} (cm)", "#it{r} (cm)", "mean distortion <d#it{z}> (cm)",
                x_offset=1.0, y_offset=1.2, z_offset=1.2, label_size=0.035)
    canvas.SetMargin(0.1, 0.15, 0.1, 0.03) # left, right, bottom, top
    canvas.SaveAs("%s/r_z_meanDistZ_colz_phi_sector0.png" % dirplots)

    t.Draw("flucSC:r:z>>htemp(65, 0, 250, 65, 83, 255)", "phi>0 && phi<3.14/9", "profcolz")
    setup_frame(None, "#it{z} (cm)", "#it{r} (cm)", "#it{#rho}_{SC} - #it{<#rho>}_{SC} (fC/cm^{3})",
                x_offset=1.0, y_offset=1.2, z_offset=1.5, label_size=0.035)
    canvas.SetMargin(0.1, 0.15, 0.1, 0.03) # left, right, bottom, top
    canvas.SaveAs("%s/flucSC_r_z_profcolz_phi_sector0.png" % dirplots)

    t.SetMarkerStyle(kDot)

    t.Draw("meanSC:z>>htemp(65, 0, 250, 20, 0.1, 0.15)", "", "profcolz")
    setup_frame(None, "#it{z} (cm)", "#it{<#rho>}_{SC} (fC/cm^{3})",
                x_offset=1.0, y_offset=1.9, label_size=0.035)
    canvas.SetMargin(0.15, 0.02, 0.1, 0.03) # left, right, bottom, top
    canvas.SaveAs("%s/meanSC_z_profcolz.png" % dirplots)

    t.Draw("meanSC-flucSC:z>>htemp(65, 0, 250, 20, 0.1, 0.2)", "", "profcolz")
    setup_frame(None, "#it{z} (cm)", "#it{#rho}_{SC} (fC/cm^{3})",
                x_offset=1.0, y_offset=1.5, label_size=0.035)
    canvas.SetMargin(0.15, 0.02, 0.1, 0.03) # left, right, bottom, top
    canvas.SaveAs("%s/randomSC_z_profcolz.png" % dirplots)

    t.SetMarkerStyle(kFullSquare)

    if draw_idc:
        t.Draw("r:z:meanCorrR", "phi>0 && phi<3.14/9", "colz")
        setup_frame(None, "#it{z} (cm)", "#it{r} (cm)", "mean correction <d#it{r}> (cm)",
                    x_offset=1.0, y_offset=1.2, z_offset=1.2, label_size=0.035)
        canvas.SetMargin(0.1, 0.15, 0.1, 0.03) # left, right, bottom, top
        canvas.SaveAs("%s/r_z_meanCorrR_colz_phi_sector0.png" % dirplots)

        t.Draw("r:z:meanCorrRPhi", "phi>0 && phi<3.14/9", "colz")
        setup_frame(None, "#it{z} (cm)", "#it{r} (cm)", "mean correction <d#it{r#varphi}> (cm)",
                    x_offset=1.0, y_offset=1.2, z_offset=1.2, label_size=0.035)
        canvas.SetMargin(0.1, 0.15, 0.1, 0.03) # left, right, bottom, top
        canvas.SaveAs("%s/r_z_meanCorrRPhi_colz_phi_sector0.png" % dirplots)

        t.Draw("r:z:meanCorrZ", "phi>0 && phi<3.14/9", "colz")
        setup_frame(None, "#it{z} (cm)", "#it{r} (cm)", "mean correction <d#it{z}> (cm)",
                    x_offset=1.0, y_offset=1.2, z_offset=1.2, label_size=0.035)
        canvas.SetMargin(0.1, 0.15, 0.1, 0.03) # left, right, bottom, top
        canvas.SaveAs("%s/r_z_meanCorrZ_colz_phi_sector0.png" % dirplots)

        t.Draw("flucanvasDIDC", "", "", 200, 0)
        setup_frame(None, "fluc 1D IDC", "entries", x_offset=1.0, y_offset=1.2, label_size=0.035)
        canvas.SetMargin(0.1, 0.15, 0.1, 0.03) # left, right, bottom, top
        canvas.SaveAs("%s/fluc_1D_IDC.png" % dirplots)


def main():
    draw_input(dirplots="idc-val-plots", draw_idc=True)

if __name__ == "__main__":
    main()
