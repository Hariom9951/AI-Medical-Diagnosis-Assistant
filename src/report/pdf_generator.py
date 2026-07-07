"""PDF Medical Report Generation Module.

Generates professional clinical reports summarizing predictions, metadata,
and class probabilities from both Image and Symptom diagnosis modes.
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.utils.logger import AppLogger

logger = AppLogger.get_logger(__name__)


class MedicalReportGenerator:
    """Generates professional medical diagnosis reports in PDF format."""

    def __init__(self, output_dir: Union[str, Path] = "docs/reports/generated_reports") -> None:
        """Initializes the report generator and ensures output directory exists.

        Args:
            output_dir: Directory where generated reports will be stored.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Style sheet initialization
        self.styles = getSampleStyleSheet()

        # Color palette
        self.primary_color = colors.HexColor("#1e3a8a")  # Deep blue / clinical theme
        self.secondary_color = colors.HexColor("#3b82f6")  # Slate blue
        self.text_color = colors.HexColor("#1f2937")  # Charcoal
        self.muted_text = colors.HexColor("#6b7280")  # Gray
        self.bg_light = colors.HexColor("#f3f4f6")  # Very light gray
        self.accent_success = colors.HexColor("#10b981")  # Emerald green
        self.accent_warning = colors.HexColor("#f59e0b")  # Amber
        self.accent_danger = colors.HexColor("#ef4444")  # Red

        self._configure_custom_styles()

    def _configure_custom_styles(self) -> None:
        """Adds custom ParagraphStyles for the medical report layout."""
        self.title_style = ParagraphStyle(
            "ReportTitle",
            parent=self.styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=self.primary_color,
            spaceAfter=5,
        )
        self.subtitle_style = ParagraphStyle(
            "ReportSubtitle",
            parent=self.styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=12,
            textColor=self.secondary_color,
            spaceAfter=15,
        )
        self.section_heading = ParagraphStyle(
            "SectionHeading",
            parent=self.styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=self.primary_color,
            spaceBefore=12,
            spaceAfter=6,
            borderPadding=2,
        )
        self.body_style = ParagraphStyle(
            "ReportBody",
            parent=self.styles["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            textColor=self.text_color,
            spaceAfter=8,
        )
        self.label_style = ParagraphStyle(
            "ReportLabel",
            parent=self.styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12,
            textColor=self.primary_color,
        )
        self.value_style = ParagraphStyle(
            "ReportValue",
            parent=self.styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=self.text_color,
        )
        self.disclaimer_style = ParagraphStyle(
            "ReportDisclaimer",
            parent=self.styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=7.5,
            leading=10,
            textColor=self.muted_text,
            alignment=1,  # Centered
        )
        self.table_header_style = ParagraphStyle(
            "TableHeader",
            parent=self.styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=11,
            textColor=colors.white,
        )

    def _draw_header_footer(self, canvas: Any, doc: SimpleDocTemplate) -> None:
        """Draws page decorations, headers, and footers."""
        canvas.saveState()

        # Top border band
        canvas.setFillColor(self.primary_color)
        canvas.rect(0, 10.85 * inch, 8.5 * inch, 0.15 * inch, stroke=0, fill=1)

        # Footer divider line
        canvas.setStrokeColor(self.bg_light)
        canvas.setLineWidth(1)
        canvas.line(0.75 * inch, 0.75 * inch, 7.75 * inch, 0.75 * inch)

        # Footer text
        canvas.setFont("Helvetica-Oblique", 7.5)
        canvas.setFillColor(self.muted_text)
        canvas.drawString(
            0.75 * inch,
            0.55 * inch,
            "Confidential - AI Medical Diagnosis Assistant - Generated Report",
        )
        canvas.drawRightString(
            7.75 * inch,
            0.55 * inch,
            f"Page {doc.page}",
        )

        canvas.restoreState()

    def generate_report(
        self,
        mode: str,
        user_input: str,
        predicted_disease: str,
        confidence: float,
        predictions: List[Dict[str, Any]],
        model_used: str,
        inference_time_ms: float,
    ) -> Path:
        """Generates a professional PDF clinical report.

        Args:
            mode: "Chest X-ray Diagnosis" or "Symptom Diagnosis".
            user_input: Free-text symptom input or uploaded image name/path.
            predicted_disease: Top predicted disease class name.
            confidence: Confidense score (softmax prob) between 0.0 and 1.0.
            predictions: List of dicts representing top-N class probabilities.
            model_used: "EfficientNet-B0" or "BioBERT".
            inference_time_ms: Total model execution time in milliseconds.

        Returns:
            Path object to the generated PDF.
        """
        report_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Clean file name safe representation
        safe_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_mode = "xray" if "x-ray" in mode.lower() else "symptoms"
        filename = f"report_{clean_mode}_{safe_time}_{report_id[:8]}.pdf"
        pdf_path = self.output_dir / filename

        # Create document template
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=letter,
            rightMargin=54,
            leftMargin=54,
            topMargin=54,
            bottomMargin=54,
        )

        story: List[Any] = []

        # 1. Main Header Title
        story.append(Paragraph("CLINICAL DIAGNOSIS REPORT", self.title_style))
        story.append(
            Paragraph(
                "AI-Assisted Diagnostic Analysis & Decision Support System", self.subtitle_style
            )
        )
        story.append(Spacer(1, 10))

        # 2. Metadata / Administrative Info (Key-Value Grid)
        meta_data = [
            [
                Paragraph("Report ID:", self.label_style),
                Paragraph(report_id, self.value_style),
                Paragraph("Date & Time:", self.label_style),
                Paragraph(timestamp, self.value_style),
            ],
            [
                Paragraph("Diagnosis Mode:", self.label_style),
                Paragraph(mode, self.value_style),
                Paragraph("Model Used:", self.label_style),
                Paragraph(model_used, self.value_style),
            ],
            [
                Paragraph("Inference Speed:", self.label_style),
                Paragraph(f"{inference_time_ms:.2f} ms", self.value_style),
                Paragraph("Status:", self.label_style),
                Paragraph("Completed", self.value_style),
            ],
        ]

        meta_table = Table(meta_data, colWidths=[1.3 * inch, 2.2 * inch, 1.2 * inch, 2.3 * inch])
        meta_table.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BACKGROUND", (0, 0), (-1, -1), self.bg_light),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
                    ("BOX", (0, 0), (-1, -1), 1, self.primary_color),
                ]
            )
        )
        story.append(meta_table)
        story.append(Spacer(1, 12))

        # 3. User Input Section
        story.append(Paragraph("Patient Case / Provided Input", self.section_heading))
        cleaned_input = user_input.strip() if user_input else "No input details provided."
        story.append(Paragraph(cleaned_input, self.body_style))
        story.append(Spacer(1, 8))

        # 4. Primary Diagnostic Result Highlight
        story.append(Paragraph("Primary Diagnostic Analysis", self.section_heading))

        conf_percentage = f"{confidence * 100:.2f}%"
        bg_color = (
            self.accent_success
            if confidence >= 0.75
            else (self.accent_warning if confidence >= 0.45 else self.accent_danger)
        )

        result_data = [
            [
                Paragraph("PREDICTED CONDITION", self.table_header_style),
                Paragraph("MODEL CONFIDENCE SCORE", self.table_header_style),
            ],
            [
                Paragraph(f"<font size=12><b>{predicted_disease}</b></font>", self.value_style),
                Paragraph(
                    f"<font color='white'><b>{conf_percentage}</b></font>", self.table_header_style
                ),
            ],
        ]

        result_table = Table(result_data, colWidths=[4.0 * inch, 3.0 * inch])
        result_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), self.primary_color),
                    ("BACKGROUND", (1, 1), (1, 1), bg_color),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(result_table)
        story.append(Spacer(1, 15))

        # 5. Top-N Probability breakdown table
        story.append(
            Paragraph("Differential Diagnosis Breakdown (Probability Index)", self.section_heading)
        )

        prob_data = [
            [
                Paragraph("Rank", self.table_header_style),
                Paragraph("Diagnostic Indication / Disease", self.table_header_style),
                Paragraph("Probability Score", self.table_header_style),
            ]
        ]

        for item in predictions:
            rank = item.get("rank", "-")
            disease = item.get("disease", item.get("disease_class", "N/A"))
            prob = item.get("confidence", item.get("probability", 0.0))
            prob_pct = f"{prob * 100:.2f}%"

            prob_data.append(
                [
                    Paragraph(str(rank), self.value_style),
                    Paragraph(disease, self.value_style),
                    Paragraph(prob_pct, self.value_style),
                ]
            )

        prob_table = Table(prob_data, colWidths=[0.8 * inch, 4.2 * inch, 2.0 * inch])
        prob_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), self.secondary_color),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, self.bg_light]),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.append(prob_table)
        story.append(Spacer(1, 20))

        # 6. Disclaimer Block (Kept together to avoid splitting)
        disclaimer_text = (
            "<b>MEDICAL DISCLAIMER:</b> This document contains analysis generated by an automated Artificial "
            "Intelligence (AI) deep learning diagnostic model. It is intended solely for general reference, research, "
            "and clinical decision support. This analysis does NOT constitute definitive medical advice, diagnostic "
            "conclusions, or a treatment plan. The prediction confidence scores represent pattern correlation levels, "
            "not clinical certainty. Any medical decisions, medication prescriptions, or diagnostic procedures must "
            "be reviewed and finalized by a licensed healthcare professional or medical doctor."
        )

        disclaimer_story = [
            Spacer(1, 10),
            Paragraph(disclaimer_text, self.disclaimer_style),
        ]
        story.append(KeepTogether(disclaimer_story))

        # Build document
        doc.build(
            story, onFirstPage=self._draw_header_footer, onLaterPages=self._draw_header_footer
        )

        logger.info("PDF medical report generated successfully: %s", pdf_path)
        return pdf_path
