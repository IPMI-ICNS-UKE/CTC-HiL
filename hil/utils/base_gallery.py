from reportlab.pdfgen import canvas
from alive_progress import alive_bar
import time
import pandas as pd
from typing import List
from typing import Optional

from hil.utils.gallery_form import create_gallery_form


def chunks(lst, n):
    """
    Yield successive n-sized chunks from lst.
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class Gallery:
    def __init__(self, output_filename='_gallery_view.pdf'):
        self.output_filename = output_filename
        pass

    def plot_gallery(
            self,
            case: str,
            proben_id: str,
            df: pd.DataFrame,
            overlay_img_path_lst: List[str],
            dapi_img_path_lst: List[str],
            ck_img_path_lst: List[str],
            cd45_img_path_lst: List[str],
            column_name_1: Optional[str],
            column_name_2: Optional[str]
    ):

        overlay_img_path_lst = list(chunks(overlay_img_path_lst, 11))
        dapi_img_path_lst = list(chunks(dapi_img_path_lst, 11))
        ck_img_path_lst = list(chunks(ck_img_path_lst, 11))
        cd45_img_path_lst = list(chunks(cd45_img_path_lst, 11))

        myfile = canvas.Canvas(case + self.output_filename)

        count = 1

        if column_name_1 is None:
            ereignis_lst = list(chunks(list(range(0, len(df))), 11))
            column_name_1 = "Ereignis"
        else:
            ereignis_lst = list(chunks(df[column_name_1].tolist(), 11))

        if column_name_2 is None:
            aufnahme_lst = list(chunks(['' for _ in range(len(df))], 11))  # empty list
            column_name_2 = ""
        else:
            aufnahme_lst = list(chunks(df[column_name_2].tolist(), 11))

        print("Creating Gallery")
        with alive_bar(len(ck_img_path_lst), force_tty=True) as bar:
            for i in range(len(ck_img_path_lst)):
                time.sleep(.005)
                create_gallery_form(
                    canvas_file=myfile,
                    case_ID=case,
                    proben_ID=proben_id,
                    ereignis=ereignis_lst[i],
                    aufnahme=aufnahme_lst[i],
                    count=count,
                    overlay_dapi_ck=overlay_img_path_lst[i],
                    dapi_img=dapi_img_path_lst[i],
                    ck_img=ck_img_path_lst[i],
                    cd45_img=cd45_img_path_lst[i],
                    column_name_1=column_name_1,
                    column_name_2=column_name_2
                )
                myfile.showPage()
                count = count + 11
                bar()
        myfile.save()
