from reportlab.lib.colors import magenta, pink, blue
from reportlab.pdfgen.canvas import Canvas
from typing import List, Union

def create_gallery_form(
    canvas_file: Canvas,
    case_ID: str,
    proben_ID: str,
    ereignis: List[Union[str, int]],
    aufnahme: List[Union[str, int]],
    count: int,
    overlay_dapi_ck: List[str],
    dapi_img: List[str],
    ck_img: List[str],
    cd45_img: List[str],
    column_name_1: str,
    column_name_2: str
) -> None:
    """
    Create a gallery form on a PDF canvas with input fields and images.

    Args:
        canvas_file (Canvas): The PDF canvas to draw on.
        case_ID (str): Identifier for the case.
        proben_ID (str): Identifier for the sample.
        ereignis (List[str]): List of events.
        aufnahme (List[str]): List of recordings.
        count (int): Counter for naming form fields.
        overlay_dapi_ck (List[str]): List of paths to overlay images (DAPI/CK-PE).
        dapi_img (List[str]): List of paths to DAPI images.
        ck_img (List[str]): List of paths to CK images.
        cd45_img (List[str]): List of paths to CD45 images.
        column_name_1 (str): Name of the first custom column.
        column_name_2 (str): Name of the second custom column.
    """
    # Constants
    TITLE_FONT = ("Helvetica-Bold", 11)
    SUBTITLE_FONT = ("Helvetica-Bold", 8)
    TEXT_FONT = ("Helvetica", 8)
    BORDER_STYLE = 'inset'
    WIDTH = 27
    HEIGHT = 48
    X_LABEL = 147
    X_COMMENT = 195
    POSITIONS = [
        (50, 810, 'Gallery'),
        (100, 790, proben_ID),
        (300, 790, case_ID),
        (490, 790, 'Non-CTC (0)/ CTC (1)')
    ]

    # Set fonts and draw static text
    canvas_file.setFont(*TITLE_FONT)
    canvas_file.drawCentredString(*POSITIONS[0])

    canvas_file.setFont(*SUBTITLE_FONT)
    for position in POSITIONS[1:]:
        canvas_file.drawCentredString(*position)

    canvas_file.setFont(*TEXT_FONT)
    column_titles = [
        (50, 760, column_name_1),
        (107, 760, column_name_2),
        (160, 760, 'Label'),
        (240, 760, 'Kommentare'),
        (341, 760, 'DAPI/CK-PE'),
        (409, 760, 'DAPI'),
        (474, 760, 'CK-PE'),
        (543, 760, 'CD45')
    ]

    for x, y, text in column_titles:
        canvas_file.drawCentredString(x, y, text)

    # Create dynamic rows for images and form fields
    top_y = 720
    row_height = 60
    form = canvas_file.acroForm

    for i in range(11):
        try:
            y_position = top_y - i * row_height
            canvas_file.drawCentredString(50, y_position, str(ereignis[i]))
            canvas_file.drawCentredString(107, y_position, str(aufnahme[i]))

            form.textfield(
                name=f'label_{count + i}',
                value="",
                fontSize=10,
                x=X_LABEL,
                y=y_position - 20,
                borderStyle=BORDER_STYLE,
                borderColor=magenta,
                fillColor=pink,
                width=WIDTH,
                height=HEIGHT,
                textColor=blue,
                forceBorder=True
            )

            form.textfield(
                name=f'comment_{count + i}',
                value="",
                fontSize=10,
                x=X_COMMENT,
                y=y_position - 20,
                borderStyle=BORDER_STYLE,
                borderColor=magenta,
                fillColor=pink,
                width=90,
                height=HEIGHT,
                textColor=blue,
                forceBorder=True
            )

            # Draw images
            canvas_file.drawImage(overlay_dapi_ck[i], 317, y_position - 20)
            canvas_file.drawImage(dapi_img[i], 385, y_position - 20)
            canvas_file.drawImage(ck_img[i], 452, y_position - 20)
            canvas_file.drawImage(cd45_img[i], 520, y_position - 20)
        except IndexError:
            break