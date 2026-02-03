from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

pipeline = ASRInferencePipeline(model_card="omniASR_LLM_Unlimited_7B_v2")
audio_files = ["/path/to/eng_audio1.flac", "/path/to/deu_audio2.wav"]
lang = ["eng_Latn", "deu_Latn"]
transcriptions = pipeline.transcribe(audio_files, lang=lang, batch_size=2)

#basic asr
  batch = Seq2SeqBatch(
      source_seqs=audio_tensor,           # [BS, T_audio, D_audio] - target audio
      source_seq_lens=audio_lengths,      # [BS] - actual audio lengths
      target_seqs=text_tensor,            # [BS, T_text] - target text tokens
      target_seq_lens=text_lengths,       # [BS] - actual text lengths
      example={}                          # Empty dict - no special fields needed
  )

#language-specific asr
  batch = Seq2SeqBatch(
      source_seqs=audio_tensor,           # [BS, T_audio, D_audio] - target audio
      source_seq_lens=audio_lengths,      # [BS] - actual audio lengths
      target_seqs=text_tensor,            # [BS, T_text] - target text tokens
      target_seq_lens=text_lengths,       # [BS] - actual text lengths
      example={
          "lang": ['mxs_Latn', ...]  # [BS] - language codes per sample
      }
  )