import argparse
import pdfkit
import markdown2

if __name__ == "__main__":
    """
    Command line script to convert a markdown file to a PDF file.
    Modified from: https://stackoverflow.com/a/75897176/21823869
    """
    parser = argparse.ArgumentParser(description="Transform a markdown report to a PDF file.")
    parser.add_argument("-wkhtml", "--wkhtmlpath",
                        type=str,
                        help='The path to the wkhtmltopdf executable.')
    parser.add_argument("-md", "--mdpath",
                        type=str,
                        help='The path to the markdown file.'
                        )
    args = parser.parse_args()

    with open(args.mdpath, "r") as file:
        markdown_text = file.read()

    # Convert Markdown to HTML with table support
    html_content = markdown2.markdown(markdown_text,
                                    extras=["tables"])

    # Add CSS for table borders
    styled_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 8px;
                border: 1px solid black;
                text-align: left;
            }}
            th {{
                background-color: #3AA270;
                color: white;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    # Convert to PDF
    config = pdfkit.configuration(wkhtmltopdf=args.wkhtmlpath)
    pdfkit.from_string(styled_html, args.mdpath[:-3] + ".pdf", configuration=config)