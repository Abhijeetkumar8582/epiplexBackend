"""
PDF Summary Service for generating incremental PDFs with images and descriptions
Uses a metadata file to track batches and builds PDF incrementally
"""
from pathlib import Path
from typing import List, Dict, Optional, Any
from uuid import UUID
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image as RLImage, KeepTogether
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from PIL import Image as PILImage
import io
import json

from app.config import settings
from app.utils.logger import logger


class PDFSummaryService:
    """Service for generating incremental PDF summaries with images"""
    
    def __init__(self):
        """Initialize PDF summary service"""
        self.page_width, self.page_height = letter
        self.image_width = 5.5 * inch
        self.image_height = 4 * inch
    
    def create_pdf(
        self,
        pdf_path: Path,
        video_name: str,
        video_file_number: Optional[str] = None
    ) -> None:
        """
        Create initial PDF file with title page
        
        Args:
            pdf_path: Path where PDF will be saved
            video_name: Name of the video
            video_file_number: Optional video file number
        """
        try:
            # Create metadata file to track batches
            metadata_path = pdf_path.with_suffix('.pdf.meta')
            metadata = {
                "video_name": video_name,
                "video_file_number": video_file_number,
                "batches": []
            }
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            # Create initial PDF with title page
            doc = SimpleDocTemplate(
                str(pdf_path),
                pagesize=letter,
                rightMargin=0.75*inch,
                leftMargin=0.75*inch,
                topMargin=0.75*inch,
                bottomMargin=0.75*inch
            )
            
            styles = getSampleStyleSheet()
            
            # Custom styles
            title_style = ParagraphStyle(
                'Title',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#1a1a1a'),
                spaceAfter=30,
                alignment=TA_CENTER,
                fontName='Helvetica-Bold'
            )
            
            subtitle_style = ParagraphStyle(
                'Subtitle',
                parent=styles['Normal'],
                fontSize=14,
                textColor=colors.HexColor('#666666'),
                spaceAfter=20,
                alignment=TA_CENTER
            )
            
            # Build title page
            story = []
            story.append(Spacer(1, 2*inch))
            story.append(Paragraph('Video Analysis Summary', title_style))
            story.append(Spacer(1, 0.3*inch))
            
            if video_file_number:
                story.append(Paragraph(f'Video File: {video_file_number}', subtitle_style))
            
            story.append(Paragraph(f'Video Name: {video_name}', subtitle_style))
            story.append(Spacer(1, 1*inch))
            story.append(Paragraph('Frame-by-Frame Analysis with Summaries', styles['Normal']))
            
            doc.build(story)
            
            logger.info("PDF created", pdf_path=str(pdf_path))
            
        except Exception as e:
            logger.error("Failed to create PDF", error=str(e), exc_info=True)
            raise
    
    def append_batch_to_pdf(
        self,
        pdf_path: Path,
        batch_number: int,
        total_batches: int,
        summary_text: str,
        frames: List[Dict[str, Any]]
    ) -> None:
        """
        Append a batch of frames and summary to existing PDF by rebuilding it
        
        Args:
            pdf_path: Path to existing PDF
            batch_number: Current batch number
            total_batches: Total number of batches
            summary_text: Summary text for this batch
            frames: List of frame dictionaries with image_path, description, timestamp, etc.
        """
        try:
            # Load metadata
            metadata_path = pdf_path.with_suffix('.pdf.meta')
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            else:
                metadata = {"batches": []}
            
            # Add this batch to metadata
            batch_data = {
                "batch_number": batch_number,
                "total_batches": total_batches,
                "summary_text": summary_text,
                "frames": [
                    {
                        "timestamp": f.get("timestamp", 0.0),
                        "frame_number": f.get("frame_number"),
                        "description": f.get("description", ""),
                        "image_path": f.get("image_path", ""),
                        "meta_tags": f.get("meta_tags", [])
                    }
                    for f in frames
                ]
            }
            metadata["batches"].append(batch_data)
            
            # Save updated metadata
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            # Rebuild entire PDF with all batches
            self._rebuild_pdf(pdf_path, metadata)
            
            logger.info("Batch appended to PDF",
                       pdf_path=str(pdf_path),
                       batch_number=batch_number,
                       frames_count=len(frames))
            
        except Exception as e:
            logger.error("Failed to append batch to PDF",
                        pdf_path=str(pdf_path),
                        batch_number=batch_number,
                        error=str(e),
                        exc_info=True)
            raise
    
    def _rebuild_pdf(self, pdf_path: Path, metadata: Dict[str, Any]) -> None:
        """Rebuild the entire PDF from metadata"""
        try:
            doc = SimpleDocTemplate(
                str(pdf_path),
                pagesize=letter,
                rightMargin=0.75*inch,
                leftMargin=0.75*inch,
                topMargin=0.75*inch,
                bottomMargin=0.75*inch
            )
            
            styles = getSampleStyleSheet()
            
            # Title page styles
            title_style = ParagraphStyle(
                'Title',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#1a1a1a'),
                spaceAfter=30,
                alignment=TA_CENTER,
                fontName='Helvetica-Bold'
            )
            
            subtitle_style = ParagraphStyle(
                'Subtitle',
                parent=styles['Normal'],
                fontSize=14,
                textColor=colors.HexColor('#666666'),
                spaceAfter=20,
                alignment=TA_CENTER
            )
            
            # Content styles
            batch_heading_style = ParagraphStyle(
                'BatchHeading',
                parent=styles['Heading2'],
                fontSize=16,
                textColor=colors.HexColor('#2c3e50'),
                spaceAfter=12,
                spaceBefore=20,
                fontName='Helvetica-Bold'
            )
            
            summary_style = ParagraphStyle(
                'Summary',
                parent=styles['Normal'],
                fontSize=11,
                textColor=colors.HexColor('#34495e'),
                spaceAfter=15,
                alignment=TA_JUSTIFY,
                leading=14
            )
            
            frame_heading_style = ParagraphStyle(
                'FrameHeading',
                parent=styles['Heading3'],
                fontSize=12,
                textColor=colors.HexColor('#3498db'),
                spaceAfter=8,
                spaceBefore=15,
                fontName='Helvetica-Bold'
            )
            
            desc_style = ParagraphStyle(
                'Description',
                parent=styles['Normal'],
                fontSize=10,
                textColor=colors.HexColor('#555555'),
                spaceAfter=10,
                alignment=TA_LEFT,
                leading=12
            )
            
            story = []
            
            # Title page
            story.append(Spacer(1, 2*inch))
            story.append(Paragraph('Video Analysis Summary', title_style))
            story.append(Spacer(1, 0.3*inch))
            
            if metadata.get("video_file_number"):
                story.append(Paragraph(f'Video File: {metadata["video_file_number"]}', subtitle_style))
            
            story.append(Paragraph(f'Video Name: {metadata.get("video_name", "Unknown")}', subtitle_style))
            story.append(Spacer(1, 1*inch))
            story.append(Paragraph('Frame-by-Frame Analysis with Summaries', styles['Normal']))
            story.append(PageBreak())
            
            # Process each batch
            for batch_data in sorted(metadata.get("batches", []), key=lambda x: x.get("batch_number", 0)):
                batch_number = batch_data.get("batch_number", 0)
                total_batches = batch_data.get("total_batches", 1)
                summary_text = batch_data.get("summary_text", "")
                frames = batch_data.get("frames", [])
                
                # Batch header
                story.append(Paragraph(
                    f"Batch {batch_number} of {total_batches}",
                    batch_heading_style
                ))
                
                # Summary section (optional - can be removed if not needed)
                if summary_text:
                    story.append(Paragraph("Summary:", frame_heading_style))
                    story.append(Paragraph(summary_text, summary_style))
                    story.append(Spacer(1, 0.2*inch))
                
                # Frames section - Steps with Images
                story.append(Paragraph("Steps with Frame Images:", frame_heading_style))
                story.append(Spacer(1, 0.1*inch))
                
                # Process each frame - show as Step 1, Step 2, etc. with images
                for idx, frame in enumerate(frames, start=1):
                    timestamp = frame.get("timestamp", 0.0)
                    frame_num = frame.get("frame_number", idx)
                    description = frame.get("description", "")
                    image_path = frame.get("image_path", "")
                    meta_tags = frame.get("meta_tags", [])
                    
                    # Step header with step number
                    step_heading = ParagraphStyle(
                        'StepHeading',
                        parent=styles['Heading2'],
                        fontSize=14,
                        textColor=colors.HexColor('#2c3e50'),
                        spaceAfter=10,
                        spaceBefore=20,
                        fontName='Helvetica-Bold'
                    )
                    
                    # Format timestamp as MM:SS
                    minutes = int(timestamp // 60)
                    seconds = int(timestamp % 60)
                    timestamp_str = f"{minutes}:{seconds:02d}"
                    
                    story.append(Paragraph(
                        f"Step {idx} - Timestamp: {timestamp_str}",
                        step_heading
                    ))
                    
                    # Frame image - show image first, then description
                    if image_path and Path(image_path).exists():
                        try:
                            img = PILImage.open(image_path)
                            # Calculate aspect ratio to maintain proportions
                            aspect_ratio = img.width / img.height
                            
                            # Set max dimensions
                            max_width = 5.5 * inch
                            max_height = 4 * inch
                            
                            if aspect_ratio > 1:
                                # Landscape
                                img_width = min(max_width, img.width / 72)
                                img_height = img_width / aspect_ratio
                            else:
                                # Portrait
                                img_height = min(max_height, img.height / 72)
                                img_width = img_height * aspect_ratio
                            
                            # Resize image maintaining aspect ratio
                            img_resized = img.resize((int(img_width * 72), int(img_height * 72)), PILImage.Resampling.LANCZOS)
                            
                            # Save to bytes
                            img_bytes = io.BytesIO()
                            img_resized.save(img_bytes, format='JPEG', quality=90)
                            img_bytes.seek(0)
                            
                            # Add image with caption
                            story.append(Paragraph(f"Image {idx}", frame_heading_style))
                            story.append(RLImage(ImageReader(img_bytes), width=img_width, height=img_height))
                            story.append(Spacer(1, 0.15*inch))
                            
                        except Exception as img_error:
                            logger.warning("Failed to add image to PDF",
                                        image_path=image_path,
                                        error=str(img_error))
                            story.append(Paragraph(f"[Image {idx} not available]", desc_style))
                    
                    # Meta tags display
                    if meta_tags and isinstance(meta_tags, list) and len(meta_tags) > 0:
                        tags_text = ", ".join([str(tag) for tag in meta_tags])
                        tag_style = ParagraphStyle(
                            'MetaTags',
                            parent=styles['Normal'],
                            fontSize=9,
                            textColor=colors.HexColor('#7f8c8d'),
                            spaceAfter=8,
                            fontName='Helvetica-Oblique'
                        )
                        story.append(Paragraph(f"Meta Tags: {tags_text}", tag_style))
                    
                    # Frame description
                    if description:
                        story.append(Paragraph(f"Description:", frame_heading_style))
                        story.append(Paragraph(description, desc_style))
                    
                    story.append(Spacer(1, 0.2*inch))
                    
                    # Add page break if not last frame in batch
                    if idx < len(frames):
                        story.append(Spacer(1, 0.1*inch))
                
                story.append(PageBreak())
            
            # Build PDF
            doc.build(story)
            
        except Exception as e:
            logger.error("Failed to rebuild PDF", error=str(e), exc_info=True)
            raise
