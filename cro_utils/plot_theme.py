"""
公式PPTテーマカラー定義
"""
import matplotlib.pyplot as plt
from cycler import cycler

# 個別カラー定数
DARK_NAVY  = "#2B336C"  # ダークネイビー
LIGHT_BLUE = "#7997F9"  # ライトブルー
PINK       = "#F7418B"  # ピンク/マゼンタ
GRAY       = "#888888"  # ニュートラルグレー
SOFT_RED   = "#FB67A0"  # ソフトレッド

# デフォルトサイクル順（axes.prop_cycle に対応）
PALETTE = [DARK_NAVY, LIGHT_BLUE, PINK, GRAY, SOFT_RED]

def apply_plot_theme():
    """matplotlib の axes.prop_cycle を公式PPTテーマカラーに設定する。"""
    plt.rcParams["axes.prop_cycle"] = cycler(color=PALETTE)
