import os

import langextract as lx


# 例は、[GoogleのNER LangExtractでエンティティ抽出をやってみた : Zenn.dev](https://zenn.dev/tfutada/articles/68dd0e19d92279) から。

# Define prompt and schema guidance
prompt = """
投薬情報を抽出してください。用量、投与経路、投与頻度を含めてください。
必ず原文のまま使用し、言い換えはしないでください。
各投薬をその属性（用量、投与経路、投与頻度）に明確に関連付けてください。
"""

examples = [
    lx.data.ExampleData(
        text="ロラゼパム 5 mg を、必要時に経口で 6 時間ごとに投与する。",
        extractions=[
            lx.data.Extraction(
                extraction_class="投薬",
                extraction_text="ロラゼパム",
                attributes={
                    "用量": "5 mg",
                    "投与経路": "経口",
                    "投与頻度": "6 時間ごと（必要時）"
                }
            )
        ]
    )
]

# Input text to process
clinical_text = "患者はアスピリン 100 mg を経口で毎日服用している。"

# Run the extraction
result = lx.extract(
    text_or_documents=clinical_text,
    prompt_description=prompt,
    examples=examples,
    model_id="gpt-4o-mini",  # auto-selects OpenAI GPT-4o
    api_key=os.environ.get("OPENAI_API_KEY"),
    fence_output=True,
    use_schema_constraints=False
)

# Save and visualize
output_file = "med_extractions_j"
lx.io.save_annotated_documents([result], output_name=f"{output_file}.jsonl", output_dir=".")

