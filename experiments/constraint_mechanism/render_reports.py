"""Render the two audited source reports as readable Chinese PDFs.

Additional dependencies: reportlab>=4,<5 and pypdf>=5,<7.
No scientific runtime is imported; figures/source reports are read as files.
"""
from pathlib import Path
import argparse,json,os
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,KeepTogether

HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--reports',type=Path,default=HERE/'reports');a=p.parse_args()
    font_path=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts/simhei.ttf'
    if font_path.exists():pdfmetrics.registerFont(TTFont('ReportCN',str(font_path)));font='ReportCN'
    else:pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'));font='STSong-Light'
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='CNTitle',fontName=font,fontSize=20,leading=28,textColor=colors.HexColor('#153b50'),spaceAfter=16,wordWrap='CJK'))
    styles.add(ParagraphStyle(name='CNHead',fontName=font,fontSize=12,leading=18,textColor=colors.HexColor('#0f766e'),spaceBefore=14,spaceAfter=8,keepWithNext=True,wordWrap='CJK'))
    styles.add(ParagraphStyle(name='CNBody',fontName=font,fontSize=9.5,leading=15,spaceAfter=8,wordWrap='CJK',splitLongWords=True,allowWidows=0,allowOrphans=0))
    styles.add(ParagraphStyle(name='CNCell',fontName=font,fontSize=7.4,leading=11,wordWrap='CJK',splitLongWords=True))
    styles.add(ParagraphStyle(name='CNCaption',fontName=font,fontSize=8,leading=12,textColor=colors.HexColor('#526575'),spaceAfter=10,wordWrap='CJK'))
    width=A4[0]-84
    def para(text,style='CNBody'):return Paragraph(escape(str(text)).replace('\n','<br/>'),styles[style])
    for path in sorted(a.reports.glob('*.json')):
        content=json.loads(path.read_text(encoding='utf-8'))
        if 'blocks' not in content:continue
        target=path.with_suffix('.pdf')
        doc=SimpleDocTemplate(str(target),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=45,bottomMargin=43,
                              title=content['title'],author='Replenishment project | reproducible experiment')
        story=[]
        for i,b in enumerate(content['blocks']):
            if b['kind']=='heading':story.append(para(b['text'],'CNTitle' if i==0 else 'CNHead'))
            elif b['kind']=='paragraph':story.append(para(b['text']))
            elif b['kind']=='table':
                n=len(b['headers'])
                weights=[1.5]+[1.]*(n-1) if n>4 else ([1.1,3.] if n==2 else [1.]*n)
                widths=[width*w/sum(weights) for w in weights]
                rows=[[para(v,'CNCell') for v in b['headers']]]+[[para(v,'CNCell') for v in row] for row in b['rows']]
                t=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
                t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#d9ecea')),
                    ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
                    ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
                    ('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#508b88')),
                    ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f4f7fa')]),
                    ('LINEBELOW',(0,1),(-1,-1),.2,colors.HexColor('#dbe3e8'))]))
                story.extend([t,Spacer(1,8)])
            else:
                source=a.reports/b['path'];iw,ih=ImageReader(str(source)).getSize();h=width*ih/iw
                story.append(KeepTogether([Image(str(source),width=width,height=h),para(b['caption'],'CNCaption')]))
        def footer(canvas,doc):
            canvas.saveState();canvas.setFont(font,8);canvas.setFillColor(colors.HexColor('#526575'))
            canvas.drawString(42,24,'补货项目 | 2026-10-16 | 真实数据 / 冻结方案 / 可复现')
            canvas.drawRightString(A4[0]-42,24,str(doc.page));canvas.restoreState()
        doc.build(story,onFirstPage=footer,onLaterPages=footer)
        from pypdf import PdfReader
        pages=PdfReader(str(target)).pages
        assert len(pages)>0 and all((page.extract_text() or '').strip() for page in pages)
        print(target.name,'pages',len(pages),'bytes',target.stat().st_size)

if __name__=='__main__':main()
