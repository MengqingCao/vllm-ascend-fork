def _forward_v1_style(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Use chunked prefill for head size 192 scenario, like deepseek
        # paged_attention_splitfuse maybe crash at such scenario.
        # TODO: vanilla path will be removed after the kernel support
        # head_size 192 scenario.
        if self.head_size == 192:
            cu_seqlen_q = [0] + attn_metadata.query_lens.tolist()
            cu_seqlen_k = [0] + attn_metadata.seq_lens.tolist()
            cu_seqlen_q = torch.tensor(cu_seqlen_q, device=query.device)
            cu_seqlen_k = torch.tensor(cu_seqlen_k, device=query.device)
            cu_seqlen_q = torch.cumsum(cu_seqlen_q, dim=0)
            cu_seqlen_k = torch.cumsum(cu_seqlen_k, dim=0)
            max_seqlen_q = torch.max(attn_metadata.query_lens)
            max_seqlen_k = torch.max(attn_metadata.seq_lens)
            vanilla_chunked_prefill(output, query, self.key_cache,
                                    self.value_cache,
                                    attn_metadata.block_tables, cu_seqlen_q,
                                    cu_seqlen_k, max_seqlen_q, max_seqlen_k,
                                    self.scale, None, True)
            return output

        # Use paged attention.
        assert attn_metadata is not None
        assert attn_metadata.attn_mask is not None

        if is_310p():
            # Do reformat in case of broadcasted tensors.
            attn_metadata.attn_mask = \
                torch_npu.npu_format_cast(attn_metadata.attn_mask.contiguous(),
                                          ACL_FORMAT_FRACTAL_NZ)
            attn_metadata.seq_lens = \
                attn_metadata.seq_lens.to(device=query.device)
        num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
        '''
        max_block_id = 0
        if hasattr(attn_metadata, 'block_tables') and attn_metadata.block_tables is not None:
            if attn_metadata.block_tables.numel() > 0:
                max_block_id = int(torch.max(attn_metadata.block_tables).item())
        print("num_block:", num_block, "max_block_id:", max_block_id)
        '''
        key = self.key_cache.view(  # type: ignore
            num_block, block_size, -1)
        value = self.value_cache.view(  # type: ignore
            num_block, block_size, -1)
        bt = getattr(attn_metadata, "block_tables", None)
        '''
        if bt is None:
            print("block_tables is None")
        else:
            # 确保是在 cpu 上检查
            bt_cpu = bt.detach().cpu() if isinstance(bt, torch.Tensor) else torch.tensor(bt)
            bt_min = int(bt_cpu.min().item()) if bt_cpu.numel()>0 else None
            bt_max = int(bt_cpu.max().item()) if bt_cpu.numel()>0 else None
            print("block_tables type:", type(bt), " shape:", getattr(bt_cpu, "shape", None))
            print("block_tables min:", bt_min, " max:", bt_max)
            # 展示前 100 个唯一 id（如果太多只看前面）
            try:
                unique_ids = torch.unique(bt_cpu).tolist()
            except Exception:
                unique_ids = list(dict.fromkeys(bt_cpu.reshape(-1).tolist()))
            print("unique block ids sample (first 100):", unique_ids[:100])
            # 检查是否存在越界 id（>= key_cache first-dim）
            key_first_dim = int(self.key_cache.shape[0])
            out_of_bound_mask = (bt_cpu.reshape(-1) >= key_first_dim)
            any_oob = bool(out_of_bound_mask.any().item()) if bt_cpu.numel()>0 else False
            print("key_cache first-dim:", key_first_dim, " any block id >= key_cache.first_dim?:", any_oob)
            if any_oob:
                # 打印越界的几个 id 和它们的位置（行, col）
                vals = bt_cpu.reshape(bt_cpu.shape[0], -1)
                oob_positions = (vals >= key_first_dim).nonzero(as_tuple=False)
                print("first 20 out-of-bound positions (row, col) and their ids:")
                for idx in range(min(20, oob_positions.size(0))):
                    r, c = int(oob_positions[idx,0]), int(oob_positions[idx,1])
                    print(f"  row {r} col {c} id={int(vals[r, c].item())}")
            # 打印 attn_metadata 的类型和关键字段，帮助追溯来源
            try:
                print("attn_metadata type:", type(attn_metadata))
                if hasattr(attn_metadata, "__class__"):
                    print("attn_metadata class name:", attn_metadata.__class__.__name__)
                # 若有 __dict__，打印 keys（不要把内容贴全）
                if hasattr(attn_metadata, "__dict__"):
                    print("attn_metadata __dict__ keys:", list(attn_metadata.__dict__.keys()))
            except Exception as e:
                print("failed printing attn_metadata meta:", e)
        '''
        bt = getattr(attn_metadata, "block_tables", None)
        bt_cpu = bt.detach().cpu() if isinstance(bt, torch.Tensor) else torch.tensor(bt)
        bt_max = int(bt_cpu.max().item()) if bt_cpu.numel()>0 else None
        num_block = int(self.key_cache.shape[0])
        print("=== MORE DEBUG ===")
        print("bt.shape:", bt_cpu.shape, " bt_min/max:", (int(bt_cpu.min().item()), bt_max) if bt_cpu.numel()>0 else None)
        print("key_cache first-dim (num_block):", num_block)
        print("bt_max+1:", (bt_max+1) if bt_max is not None else None, " bt_max+2:", (bt_max+2) if bt_max is not None else None)
        print("attn_metadata.num_prefills:", getattr(attn_metadata, "num_prefills", None))
        print("attn_metadata.num_actual_tokens:", getattr(attn_metadata, "num_actual_tokens", None))
        print("attn_metadata.num_prefills type:", type(getattr(attn_metadata, "num_prefills", None)))
        print("actual_seq_lengths_q length:", len(getattr(attn_metadata, "actual_seq_lengths_q", [])))
        print("seq_lens_list length:", len(getattr(attn_metadata, "seq_lens_list", [])))
        print("block_table rows reported (len bt):", bt_cpu.shape[0])
        # # 列出包含 bt_max 的行号（有助于看是哪一条请求）
        rows_with_max = (bt_cpu == bt_max).any(dim=1).nonzero(as_tuple=False)
        print("rows_with_max count:", rows_with_max.size(0))
        for i in range(min(10, rows_with_max.size(0))):
            r = int(rows_with_max[i,0])
            print(f" row {r} sample cols 0..15:", bt_cpu[r,:16].tolist())
        print("=== END DEBUG ===")
        try:
            output, _ = torch_npu.npu_fused_infer_attention_score(
                query=query,
                key=key,
                value=value,
                atten_mask=attn_metadata.attn_mask,
                block_table=attn_metadata.block_tables,
                input_layout="TND",
                block_size=block_size,
                actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
                actual_seq_lengths_kv=attn_metadata.seq_lens_list,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale=self.scale,
                sparse_mode=3,
            )
        except RuntimeError as e:
            print("[ERROR] npu_fused_infer_attention_score failed:", e)
            print("[INFO] dumping all inputs again for postmortem...")
            safe_summary("query", query)
            safe_summary("key", key)
            safe_summary("value", value)
            safe_summary("block_size", block_size)
            print("actual_seq_lengths_q length:", len(attn_metadata.actual_seq_lengths_q))
            print("seq_lens_list length:", len(attn_metadata.seq_lens_list))
            print("num_heads:", self.num_heads)
            print("num_kv_heads:", self.num_kv_heads)
            print("block_table:", len(attn_metadata.block_tables))
            safe_summary("attn_metadata.block_tables", getattr(attn_metadata, "block_tables", None))
            raise

        return output