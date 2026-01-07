"""
Training Monitor - Real-time monitoring of XR2Text training
Shows GPU stats, progress, and estimated completion time
"""

import os
import time
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

def get_gpu_stats():
    """Get current GPU utilization and temperature"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            stats = result.stdout.strip().split(',')
            return {
                'gpu_util': float(stats[0]),
                'temp': float(stats[1]),
                'mem_used': float(stats[2]),
                'mem_total': float(stats[3])
            }
    except:
        pass
    return None

def parse_training_log():
    """Parse training history CSV to show progress"""
    history_path = Path('../data/statistics/training_history.csv')
    if history_path.exists():
        try:
            with open(history_path, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    last_line = lines[-1].strip().split(',')
                    return {
                        'current_epoch': int(last_line[0]) if len(last_line) > 0 else 0,
                        'bleu4': float(last_line[4]) if len(last_line) > 4 else 0,
                        'rougel': float(last_line[7]) if len(last_line) > 7 else 0
                    }
        except:
            pass
    return None

def monitor():
    """Main monitoring loop"""
    print("=" * 60)
    print("  XR2Text Training Monitor")
    print("  Press Ctrl+C to exit (training will continue)")
    print("=" * 60)
    print()

    start_time = datetime.now()
    total_epochs = 50

    while True:
        os.system('cls' if os.name == 'nt' else 'clear')

        print("=" * 60)
        print(f"  Training Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        print()

        # GPU Stats
        gpu_stats = get_gpu_stats()
        if gpu_stats:
            print(f"🖥️  GPU Status:")
            print(f"   Utilization: {gpu_stats['gpu_util']:.1f}%")
            print(f"   Temperature: {gpu_stats['temp']:.1f}°C")
            print(f"   Memory: {gpu_stats['mem_used']:.0f}/{gpu_stats['mem_total']:.0f} MB ({gpu_stats['mem_used']/gpu_stats['mem_total']*100:.1f}%)")

            # Temperature warning
            if gpu_stats['temp'] > 80:
                print(f"   ⚠️  WARNING: High temperature! Ensure good cooling.")
            print()

        # Training Progress
        progress = parse_training_log()
        if progress:
            current_epoch = progress['current_epoch']
            percent_complete = (current_epoch / total_epochs) * 100

            print(f"📊 Training Progress:")
            print(f"   Epoch: {current_epoch}/{total_epochs} ({percent_complete:.1f}%)")
            print(f"   BLEU-4: {progress['bleu4']:.4f}")
            print(f"   ROUGE-L: {progress['rougel']:.4f}")

            # Progress bar
            bar_length = 40
            filled = int(bar_length * current_epoch / total_epochs)
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f"   [{bar}]")

            # Time estimate
            elapsed = datetime.now() - start_time
            if current_epoch > 0:
                time_per_epoch = elapsed / current_epoch
                remaining_epochs = total_epochs - current_epoch
                eta = datetime.now() + (time_per_epoch * remaining_epochs)

                print(f"\n⏱️  Time Info:")
                print(f"   Elapsed: {str(elapsed).split('.')[0]}")
                print(f"   Avg per epoch: {str(time_per_epoch).split('.')[0]}")
                print(f"   ETA: {eta.strftime('%Y-%m-%d %H:%M:%S')}")
            print()
        else:
            print("📊 Training Progress:")
            print("   Waiting for training to start...")
            print()

        # Checkpoints
        checkpoint_dir = Path('../checkpoints')
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob('*.pt'))
            if checkpoints:
                print(f"💾 Checkpoints: {len(checkpoints)} saved")
                latest = max(checkpoints, key=os.path.getmtime)
                print(f"   Latest: {latest.name}")
                print()

        print("=" * 60)
        print("Refreshing in 30 seconds... (Ctrl+C to exit)")
        print("=" * 60)

        time.sleep(30)

if __name__ == '__main__':
    try:
        monitor()
    except KeyboardInterrupt:
        print("\n\nMonitor stopped. Training continues in background.")
