"""Build hourly per-device training CSVs from cleaned alarm events (stdlib only)."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEVELS = ['w_level1', 'w_level2', 'w_level3', 'w_level4']
LEVEL_MAP = dict(zip(['提示', '次要', '重要', '紧急'], range(4)))


def write_csv(path, columns, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def build(source, output):
    files = sorted(source.glob('*.csv'), key=lambda p: int(p.name.split('_')[0][2:]))
    if not files:
        raise ValueError(f'No input CSVs: {source}')
    # Validate every source before producing any outputs. No event is silently dropped.
    devices = []
    all_types = set()
    for path in files:
        events = []
        with path.open(encoding='utf-8-sig', newline='') as stream:
            for line, row in enumerate(csv.DictReader(stream), 2):
                try:
                    timestamp = datetime.strptime(row['发生时间'], '%m/%d/%Y %H:%M:%S')
                    level = LEVEL_MAP[row['级别']]
                    repair = float(row['Repair_Interval'])
                    assert math.isfinite(repair)
                    assert row['告警源'] == path.name.split('_')[0]
                    alarm_type = row['告警类型']
                    assert alarm_type
                except (ValueError, KeyError, AssertionError) as exc:
                    raise ValueError(f'{path}:{line}: invalid event') from exc
                events.append((timestamp.replace(minute=0, second=0, microsecond=0), level, alarm_type, repair))
                all_types.add(alarm_type)
        if not events:
            raise ValueError(f'Empty device file: {path}')
        devices.append((path, events))
    types = sorted(all_types)
    for subdir in ['four_levels', 'rich']:
        (output / subdir).mkdir(parents=True, exist_ok=True)
    audit = []
    for path, events in devices:
        buckets = defaultdict(list)
        expected = Counter()
        for event in events:
            buckets[event[0]].append(event)
            expected[event[1]] += 1
        start, end = min(buckets), max(buckets)
        rows, rich_rows = [], []
        hour = start
        while hour <= end:
            batch = buckets.get(hour, [])
            counts = Counter(event[1] for event in batch)
            type_counts = Counter(event[2] for event in batch)
            row = [hour.strftime('%Y-%m-%d %H:%M:%S')] + [counts[i] for i in range(4)]
            rows.append(row)
            rich_rows.append(row + [type_counts[t] for t in types] + [sum(e[3] for e in batch)/len(batch) if batch else 0.0])
            hour += timedelta(hours=1)
        device = path.name.split('_')[0]
        four_path = output / 'four_levels' / f'{device}_cleaned.csv'
        rich_path = output / 'rich' / path.name
        write_csv(four_path, ['date'] + LEVELS, rows)
        write_csv(rich_path, ['date'] + LEVELS + ['type_' + t for t in types] + ['avg_repair_time'], rich_rows)
        # Read actual output back; check time continuity, shape and event conservation.
        for target, width in [(four_path, 5), (rich_path, 6 + len(types))]:
            with target.open(encoding='utf-8', newline='') as stream:
                saved = list(csv.reader(stream))[1:]
            assert len(saved) == len(rows)
            assert all(len(row) == width for row in saved)
            assert all(datetime.fromisoformat(row[0]) == start + timedelta(hours=i) for i, row in enumerate(saved))
            assert all(sum(int(row[j+1]) for row in saved) == expected[j] for j in range(4))
            assert all(math.isfinite(float(value)) for row in saved for value in row[1:])
        assert sum(sum(row[5:5+len(types)]) for row in rich_rows) == len(events)
        audit.append([device, path.name, len(events), len(rows), rows[0][0], rows[-1][0], len(rows)-len(buckets)] + [expected[i] for i in range(4)] + [hashlib.sha256(path.read_bytes()).hexdigest()])
    write_csv(output / 'device_audit.csv', ['device', 'source_file', 'source_events', 'hourly_rows', 'start', 'end', 'zero_hours'] + LEVELS + ['source_sha256'], audit)
    summary = dict(source=str(source.resolve()), devices=len(devices), source_events=sum(len(e) for _, e in devices), hourly_rows_per_version=sum(r[3] for r in audit), frequency='1h', time_range='per-device first through last event hour, inclusive', alarm_types=types, validation='all files read back; hourly continuity, finite numeric values and per-level event totals verified')
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output / 'README.md').write_text('''# 全设备清洗数据训练 CSV

- `four_levels/设备N_cleaned.csv`：date,w_level1,w_level2,w_level3,w_level4，与当前四级输入训练加载器兼容。
- `rich/设备N_For_Machine_Learning.csv`：四级计数、统一的告警类型计数列、avg_repair_time，与原 SAFE 扩展特征格式一致。
- 按发生时间向下取整到整点，统计每个左闭右开的小时；提示/次要/重要/紧急对应级别1/2/3/4。每台设备从首次至末次告警所在小时连续输出，无事件小时补零；不删除重复告警，不再次删除长零区间。没有观测覆盖信息，补零不代表已确认设备在线；不向首末事件范围外延伸。
- 修复时长沿用原始 Repair_Interval 的单位，取该小时事件均值。该字段可能含预测时点尚未完成的修复信息；当前四级故障预测请使用 four_levels。
- 所有设备使用统一列顺序，date不计入特征维度。保留完整序列，训练/验证/测试由训练加载器按时间切分，不在此处生成滑窗或未来标签。
- device_audit.csv 记录逐设备事件数、小时数、各级总数、源文件SHA256；summary.json 为全量汇总。
- 本目录来源于 fault_raw/raw_data_cleaned，不等同于旧 fault_selected_cleaned 的长零区间清洗口径。

重新生成：在项目根目录运行 `python3 scripts/tools/build_all_cleaned_training.py`。
''', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT / 'dataset/fault_raw/raw_data_cleaned')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'dataset/fault_raw/processed_all_cleaned')
    args = parser.parse_args()
    build(args.input_dir, args.output_dir)
