# pdf_reader_tool.py
import PyPDF2

class PDFReaderTool:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_pdf(self):
        try:
            with open(self.file_path, 'rb') as file:
                reader = PyPDF2.PdfFileReader(file)
                content = []
                for page_num in range(reader.numPages):
                    page = reader.getPage(page_num)
                    content.append(page.extractText())
                return "\n".join(content)
        except Exception as e:
            return str(e)

# Example usage
if __name__ == "__main__":
    pdf_tool = PDFReaderTool("sample.pdf")
    print(pdf_tool.read_pdf())
