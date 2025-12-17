import os
from pdf2image import convert_from_path, exceptions

# Assuming app.log_config.logger and PROJECT_ROOT are defined as in your original snippet

# --- Minimal Stand-ins for missing context ---
class Logger:
    def info(self, msg): print(f"INFO: {msg}")
    def error(self, msg): print(f"ERROR: {msg}")
logger = Logger()
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ---------------------------------------------


def convert_pdf_to_png(local_file_path: str, file_name: str) -> str:
    """
    Converts a multi-page PDF file into a series of PNG images (one per page)
    using the pdf2image library (which relies on Poppler).

    Args:
        local_file_path: The absolute path to the input PDF file.
        file_name: A base name to use for the temporary output directory.

    Returns:
        The absolute path to the directory where the PNG files are saved.
    """
    
    # NOTE for Windows Users: If Poppler is not in your PATH, 
    # you might need to specify the path to its bin folder here:
    # POPPLER_PATH = r"C:\path\to\poppler\bin" 

    try:
        logger.info(f"Local PDF file path: {local_file_path} - {file_name}")
        
        # Define the output directory structure, similar to your existing logic
        output_dir = os.path.join(PROJECT_ROOT, "tempfiles", file_name, "pages")
        logger.info(f"Output directory for PNGs: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        # 1. Convert PDF pages to a list of Pillow Image objects
        logger.info("Starting PDF conversion using pdf2image...")
        
        # 'dpi=200' is a good balance for quality and file size.
        # 'thread_count' can speed up conversion of large PDFs.
        # Do NOT pass `output_folder` here. When `output_folder` is provided,
        # pdf2image (pdftoppm) will write image files to disk and then
        # return PIL images — calling `page.save(...)` afterwards results in
        # duplicate files. Instead, let pdf2image return PIL images only and
        # save them once below.
        pages = convert_from_path(
            pdf_path=local_file_path,
            dpi=200,
            fmt="png",
            thread_count=4,
            # poppler_path=POPPLER_PATH # Uncomment and set this if required on Windows
        )

        # 2. Save each Pillow Image object to the output directory
        for i, page in enumerate(pages):
            # Create a consistent file name for each page
            page_output_path = os.path.join(output_dir, f"page_{i + 1}.png")
            page.save(page_output_path, "PNG")
            logger.info(f"Saved page {i + 1} to: {page_output_path}")

        logger.info(f"Successfully converted {len(pages)} pages.")
        return output_dir
        
    except exceptions.PDFPageCountError:
        logger.error(f"Failed to read PDF pages from {local_file_path}. Is the file valid?")
        raise RuntimeError("The input file is not a valid PDF or is corrupted.")
    except exceptions.PopplerNotInstalledError:
        # This is a critical error for pdf2image
        logger.error("Poppler utility is not installed. Please install poppler-utils/poppler.")
        raise RuntimeError("Poppler (non-Python dependency) is missing. Cannot convert PDF.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during PDF conversion: {e}")
        raise