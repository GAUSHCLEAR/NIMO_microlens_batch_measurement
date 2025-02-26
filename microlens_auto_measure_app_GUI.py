import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 使用非交互式后端
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image, ImageTk

# ===== 引用自定义处理文件 =====
# 请保证与本地 data_processing.py 和 report_processing.py 保持一致
from data_processing import *
from report_processing import *

point_per_mm = 26.058823529411764

class MicrolensMeasurementApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # 设置窗口基本属性
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")
        self.title("微透镜自动测量")
        self.geometry("1000x700")

        # session_state 存储
        self.session_state = {
            'report_plot': None,
            'report_text': None,
            'data_measure_csv': None,
            'data_measure': None
        }

        # 字段存储
        self.filename_design = None
        self.filename_measure = None
        self.point_per_mm = 26.058823529411764

        # 主容器：分为左侧的控制面板和右侧的显示面板
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True)

        self.left_frame = ctk.CTkFrame(self.main_frame, width=250)
        self.left_frame.pack(side="left", fill="y", padx=5, pady=5)

        self.right_frame = ctk.CTkFrame(self.main_frame)
        self.right_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        # 在右侧框架中放置一个Tabview，用于展示多个图
        self.tabview = ctk.CTkTabview(self.right_frame, width=800, height=600)
        self.tabview.pack(fill="both", expand=True, padx=5, pady=5)

        # 创建若干Tab，后续可以往不同Tab里添加内容
        self.tab_plot = self.tabview.add("检测定位图")
        self.tab_data_measure = self.tabview.add("测量散点图")
        self.tab_text_report = self.tabview.add("测量结果")

        # 在左侧放置交互组件：按钮、文本等
        self.create_left_frame_widgets()

        # 在右侧各tab初始化
        self.report_plot_label = None
        self.measure_plot_label = None
        self.text_box_report = None

    def create_left_frame_widgets(self):
        """在左侧的 Frame 中创建按钮和标签"""
        # 选择设计样板按钮
        self.btn_design = ctk.CTkButton(self.left_frame, text="1. 选择设计样板CSV",
                                        command=self.open_design_file)
        self.btn_design.pack(pady=(10, 5), padx=10, fill="x")

        # 选择测量数据按钮
        self.btn_measure = ctk.CTkButton(self.left_frame, text="2. 选择测量数据CSV",
                                         command=self.open_measure_file)
        self.btn_measure.pack(pady=5, padx=10, fill="x")

        # 开始测量按钮
        self.btn_start = ctk.CTkButton(self.left_frame, text="3. 开始测量",
                                       command=self.start_measure)
        self.btn_start.pack(pady=5, padx=10, fill="x")

        # 下载/保存按钮
        self.btn_save = ctk.CTkButton(self.left_frame, text="下载测量结果CSV",
                                      command=self.save_measure_csv)
        self.btn_save.pack(pady=5, padx=10, fill="x")

    def open_design_file(self):
        """选择设计样板CSV文件"""
        file_path = filedialog.askopenfilename(
            title="选择设计样板CSV文件",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if file_path:
            self.filename_design = file_path
            messagebox.showinfo("提示", f"设计样板文件已选择：{self.filename_design}")

    def open_measure_file(self):
        """选择测量数据CSV文件"""
        file_path = filedialog.askopenfilename(
            title="选择测量数据CSV文件",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if file_path:
            self.filename_measure = file_path
            messagebox.showinfo("提示", f"测量数据文件已选择：{self.filename_measure}")

    def start_measure(self):
        """点击开始测量按钮后，执行数据处理并展示结果"""
        if (self.filename_design is not None) and (self.filename_measure is not None):
            # 读取数据
            data_origin = pd.read_csv(self.filename_design)

            data = read_data(self.filename_measure)
            # 计算 point_per_mm
            # 假设此测量文件中宽度=17mm，可根据自己需求做动态调整
            self.point_per_mm = data.shape[0] / 17
            mm_per_point = 1 / self.point_per_mm
            print("测量文件名:", self.filename_measure)

            # 开始处理
            # 1. 微透镜识别与定位
            binary_image = detect_edge(data, threshold=0.7)
            microlenses, microlens_only_image = label_microlens(
                binary_image,
                min_area=15 * 15,
            )
            microlens_params = microlens_centers_radius(microlenses)

            # 2. 微透镜邻域测量
            sorted_microlens_params = calculate_each_lens_Rx(
                data, microlens_params, fix_raidus=0.6 * self.point_per_mm
            )

            # 3. 微透镜加光度测量
            sorted_microlens_params = measure_microlens_in_diamter(
                sorted_microlens_params, data, self.point_per_mm,
                diameter_list=[0.7, 0.5, 0.3, 0.1]
            )

            # 4. 微透镜对齐
            aligned_coords = align_microlens(
                sorted_microlens_params, data_origin, self.point_per_mm
            )

            # 5. 更新微透镜参数
            sorted_microlens_params = update_microlens_params_after_align(
                sorted_microlens_params, aligned_coords, data_origin
            )

            # 6. 生成报告图片
            report_plot = report_align_location(data_origin, data, aligned_coords, sorted_microlens_params)
            self.session_state["report_plot"] = report_plot

            # 7. 环区分析并生成报告文字
            ring_param_list = analysis_ring(sorted_microlens_params)
            report_text = generate_ring_report(ring_param_list)
            self.session_state["report_text"] = report_text

            # 8. 生成测量csv和对应的散点图
            data_measure_csv, data_measure = generate_report_csv(
                data_origin, sorted_microlens_params
            )
            self.session_state["data_measure_csv"] = data_measure_csv
            self.session_state["data_measure"] = data_measure

            fig_data_measure = report_data_measure(data_measure)

            # 在GUI中显示图像和文本
            self.show_plot_in_tab(report_plot, self.tab_plot, mode="report_plot")
            self.show_plot_in_tab(fig_data_measure, self.tab_data_measure, mode="measure_plot")
            self.show_text_in_tab(report_text, self.tab_text_report)

            messagebox.showinfo("提示", "测量完成！")
        else:
            messagebox.showwarning("警告", "请先选择设计样板文件与测量数据文件！")

    def show_plot_in_tab(self, figure, tab_frame, mode="report_plot"):
        """
        将 matplotlib Figure 转为图像后显示在指定 tab Frame 中
        mode: 用来区分是report图，还是data measure图
        """
        # 清空原有内容（若重复测量）
        for child in tab_frame.winfo_children():
            child.destroy()

        # 将 Figure 保存到内存字节流
        buf = BytesIO()
        figure.savefig(buf, format='png', dpi=100)
        buf.seek(0)

        # 转为 PIL Image
        pil_image = Image.open(buf)
        img_w, img_h = pil_image.size
        # 如果图太大，可以进行适当缩放
        scale_factor = 1.0
        max_size = 600  # 最大显示尺寸
        if img_w > max_size or img_h > max_size:
            scale_factor = max_size / float(max(img_w, img_h))
            new_size = (int(img_w * scale_factor), int(img_h * scale_factor))
            pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)

        # 转为 ImageTk
        tk_image = ImageTk.PhotoImage(pil_image)

        # 创建一个 Label 来展示图片
        label = ctk.CTkLabel(tab_frame, text="", image=tk_image)
        label.image = tk_image
        label.pack(pady=10)

        # 存起来用于后续可能的刷新
        if mode == "report_plot":
            self.report_plot_label = label
        else:
            self.measure_plot_label = label

    def show_text_in_tab(self, text_content, tab_frame):
        """
        在指定 TabFrame 中显示文本报告
        """
        # 清空原有内容
        for child in tab_frame.winfo_children():
            child.destroy()

        # Textbox 或 ScrolledText
        text_box = ctk.CTkTextbox(tab_frame, width=780, height=550)
        text_box.insert("0.0", text_content)
        text_box.configure(state="disabled")
        text_box.pack(fill="both", expand=True, padx=10, pady=10)

        self.text_box_report = text_box

    def save_measure_csv(self):
        """
        将测量结果另存为 CSV
        """
        if self.session_state["data_measure_csv"] is not None:
            save_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
            )
            if save_path:
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(self.session_state["data_measure_csv"])
                messagebox.showinfo("提示", f"测量结果已保存：{save_path}")
        else:
            messagebox.showwarning("警告", "尚无测量结果，请先完成测量！")


if __name__ == "__main__":
    app = MicrolensMeasurementApp()
    app.mainloop()
