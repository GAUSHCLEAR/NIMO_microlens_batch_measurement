import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
warnings.filterwarnings('ignore')

from data_processing import *
from report_processing import *

import matplotlib.cm as cm
import matplotlib.colors as mcolors

class MicrolensMeasurementApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("微透镜测量")
        self.geometry("1200x800")
        
        # 设置关闭协议，确保程序退出
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 文件路径
        self.filename = None
        
        # 加光度范围参数，默认 [3,10] 间隔 0.5
        self.min_power_var = tk.DoubleVar(value=3.0)
        self.max_power_var = tk.DoubleVar(value=10.0)
        self.interval_var = tk.DoubleVar(value=0.5)
        self.add_power_list = []   # 动态生成的加光度列表
        self.addpower_vars = {}    # 格式：{加光度: {"color": tk.StringVar, "value": tk.DoubleVar, "button": 按钮}}
        
        self.ring_choice = tk.StringVar(value="内圈")
        self.i_center_y = tk.DoubleVar(value=17/2)
        self.i_center_x = tk.DoubleVar(value=17/2)
        # 移除允差输入，允差将用间隔的一半代替
        self.diameter = tk.DoubleVar(value=0.7)
        self.ring_number_list_str = tk.StringVar(value="0,1,3-10")
        
        self.canvas = None  # 用于显示主图形
        
        self.create_widgets()
        # 首次启动自动生成加光列表
        self.generate_addpower_list()
    
    def create_widgets(self):
        # ===== 左侧参数输入区域（放入 Canvas 内，添加滚动条） =====
        left_frame = tk.Frame(self)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        left_canvas = tk.Canvas(left_frame, borderwidth=0, width=350)
        left_scrollbar = tk.Scrollbar(left_frame, orient="vertical", command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_scrollbar.set)
        left_scrollbar.pack(side="right", fill="y")
        left_canvas.pack(side="left", fill="both", expand=True)
        
        self.options_frame = tk.Frame(left_canvas)
        left_canvas.create_window((0,0), window=self.options_frame, anchor="nw")
        self.options_frame.bind("<Configure>", lambda event: left_canvas.configure(scrollregion=left_canvas.bbox("all")))
        
        # 1. 文件选择
        file_frame = tk.LabelFrame(self.options_frame, text="1. 选择CSV文件", padx=5, pady=5)
        file_frame.pack(fill=tk.X, pady=5)
        self.file_label = tk.Label(file_frame, text="未选择文件")
        self.file_label.pack(side=tk.LEFT, padx=5)
        file_button = tk.Button(file_frame, text="选择文件", command=self.select_file)
        file_button.pack(side=tk.RIGHT, padx=5)
        
        # 2. 内圈/中圈/外圈选择
        ring_frame = tk.LabelFrame(self.options_frame, text="2. 选择内圈/中圈/外圈", padx=5, pady=5)
        ring_frame.pack(fill=tk.X, pady=5)
        for text in ["内圈", "中圈", "外圈"]:
            rb = tk.Radiobutton(ring_frame, text=text, variable=self.ring_choice, value=text, command=self.update_center_defaults)
            rb.pack(anchor=tk.W)
        
        # 2.1/2.2 镜片中心坐标设置
        center_frame = tk.LabelFrame(self.options_frame, text="2.1/2.2 设定镜片中心坐标", padx=5, pady=5)
        center_frame.pack(fill=tk.X, pady=5)
        tk.Label(center_frame, text="横坐标:").grid(row=0, column=0, sticky=tk.W)
        self.entry_center_y = tk.Entry(center_frame, textvariable=self.i_center_y)
        self.entry_center_y.grid(row=0, column=1)
        tk.Label(center_frame, text="纵坐标:").grid(row=1, column=0, sticky=tk.W)
        self.entry_center_x = tk.Entry(center_frame, textvariable=self.i_center_x)
        self.entry_center_x.grid(row=1, column=1)
        
        # 3. 测量参数（仅保留测量直径）
        measure_frame = tk.LabelFrame(self.options_frame, text="3. 测量参数", padx=5, pady=5)
        measure_frame.pack(fill=tk.X, pady=5)
        tk.Label(measure_frame, text="测量直径:").grid(row=0, column=0, sticky=tk.W)
        self.entry_diameter = tk.Entry(measure_frame, textvariable=self.diameter)
        self.entry_diameter.grid(row=0, column=1)
        
        # 4. 加光度范围设定
        power_range_frame = tk.LabelFrame(self.options_frame, text="4. 加光度范围设定", padx=5, pady=5)
        power_range_frame.pack(fill=tk.X, pady=5)
        tk.Label(power_range_frame, text="最小值:").grid(row=0, column=0, sticky=tk.W)
        tk.Entry(power_range_frame, textvariable=self.min_power_var).grid(row=0, column=1)
        tk.Label(power_range_frame, text="最大值:").grid(row=1, column=0, sticky=tk.W)
        tk.Entry(power_range_frame, textvariable=self.max_power_var).grid(row=1, column=1)
        tk.Label(power_range_frame, text="间隔:").grid(row=2, column=0, sticky=tk.W)
        tk.Entry(power_range_frame, textvariable=self.interval_var).grid(row=2, column=1)
        tk.Button(power_range_frame, text="生成加光列表", command=self.generate_addpower_list).grid(row=3, column=0, columnspan=2, pady=5)
        
        # 5. 加光颜色设定（折叠区域）
        self.addpower_collapsible_frame = tk.LabelFrame(self.options_frame, text="5. 加光颜色设定", padx=5, pady=5)
        self.addpower_collapsible_frame.pack(fill=tk.X, pady=5)
        self.toggle_btn = tk.Button(self.addpower_collapsible_frame, text="折叠", command=self.toggle_addpower)
        self.toggle_btn.pack(anchor=tk.W)
        self.addpower_container = tk.Frame(self.addpower_collapsible_frame)
        self.addpower_container.pack(fill=tk.X)
        self.addpower_visible = True
        
        # 6. 微透镜列表输入
        ring_number_frame = tk.LabelFrame(self.options_frame, text="6. 一组微透镜测量", padx=5, pady=5)
        ring_number_frame.pack(fill=tk.X, pady=5)
        tk.Label(ring_number_frame, text="微透镜列表 (例如: 0,1,3-10):").pack(anchor=tk.W)
        self.entry_ring_number = tk.Entry(ring_number_frame, textvariable=self.ring_number_list_str)
        self.entry_ring_number.pack(fill=tk.X)
        
        # “确定”按钮
        confirm_button = tk.Button(self.options_frame, text="确定", command=self.run_measurement)
        confirm_button.pack(pady=10)
        
        # ===== 右侧结果显示区域（放入 Canvas 内，添加滚动条） =====
        right_frame = tk.Frame(self)
        right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        
        right_canvas = tk.Canvas(right_frame, borderwidth=0)
        right_scrollbar = tk.Scrollbar(right_frame, orient="vertical", command=right_canvas.yview)
        right_canvas.configure(yscrollcommand=right_scrollbar.set)
        right_scrollbar.pack(side="right", fill="y")
        right_canvas.pack(side="left", fill="both", expand=True)
        
        self.results_frame = tk.Frame(right_canvas)
        right_canvas.create_window((0,0), window=self.results_frame, anchor="nw")
        self.results_frame.bind("<Configure>", lambda event: right_canvas.configure(scrollregion=right_canvas.bbox("all")))
        
        report_frame = tk.LabelFrame(self.results_frame, text="报告", padx=5, pady=5)
        report_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.report_text = tk.Text(report_frame, wrap=tk.WORD, height=10)
        self.report_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        report_scroll = tk.Scrollbar(report_frame, command=self.report_text.yview)
        self.report_text.configure(yscrollcommand=report_scroll.set)
        report_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.figure_frame = tk.LabelFrame(self.results_frame, text="图形展示", padx=5, pady=5)
        self.figure_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 新增警告微透镜显示区域（嵌入在主窗口内）
        self.warning_frame = tk.LabelFrame(self.results_frame, text="警告微透镜", padx=5, pady=5)
        self.warning_frame.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def toggle_addpower(self):
        if self.addpower_visible:
            self.addpower_container.pack_forget()
            self.toggle_btn.config(text="展开")
            self.addpower_visible = False
        else:
            self.addpower_container.pack(fill=tk.X)
            self.toggle_btn.config(text="折叠")
            self.addpower_visible = True
    
    def update_center_defaults(self):
        choice = self.ring_choice.get()
        if choice == '内圈':
            self.i_center_x.set(17/2)
            self.i_center_y.set(17/2)
        elif choice == '中圈':
            self.i_center_x.set(17/2)
            self.i_center_y.set(17/2+11)
        elif choice == '外圈':
            self.i_center_x.set(17/2)
            self.i_center_y.set(17/2+16)
    
    def choose_color_custom(self, addpower, color_var, button):
        color_code = colorchooser.askcolor(title="选择颜色")[1]
        if color_code:
            color_var.set(color_code)
            button.configure(bg=color_code)
    
    def select_file(self):
        filename = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if filename:
            self.filename = filename
            self.file_label.config(text=os.path.basename(filename))
    
    def generate_addpower_list(self):
        try:
            min_val = self.min_power_var.get()
            max_val = self.max_power_var.get()
            interval = self.interval_var.get()
            if interval <= 0:
                messagebox.showerror("错误", "间隔必须大于0")
                return
            num = int(round((max_val - min_val) / interval)) + 1
            self.add_power_list = [round(min_val + i * interval, 2) for i in range(num)]
            cmap = cm.get_cmap("rainbow", len(self.add_power_list))
            orig_color_list = [mcolors.to_hex(cmap(i)) for i in range(len(self.add_power_list))]
            
            # 重排颜色列表，交叉排列（例如：[0, -1, 1, -2, ...]）
            def reorder_colors(color_list):
                new_colors = []
                left, right = 0, len(color_list) - 1
                toggle = True
                while left <= right:
                    if toggle:
                        new_colors.append(color_list[left])
                        left += 1
                    else:
                        new_colors.append(color_list[right])
                        right -= 1
                    toggle = not toggle
                return new_colors
            
            self.addpower_color_list = reorder_colors(orig_color_list)
            
            # 清空原有加光列表区内容
            for widget in self.addpower_container.winfo_children():
                widget.destroy()
            self.addpower_vars = {}
            for idx, addpower in enumerate(self.add_power_list):
                frame = tk.Frame(self.addpower_container)
                frame.pack(fill=tk.X, pady=2)
                tk.Label(frame, text=f"加光度 {addpower}:").pack(side=tk.LEFT)
                color_var = tk.StringVar(value=self.addpower_color_list[idx])
                btn = tk.Button(frame, text="选择颜色", bg=color_var.get())
                btn.configure(command=lambda a=addpower, cv=color_var, b=btn: self.choose_color_custom(a, cv, b))
                btn.pack(side=tk.LEFT, padx=5)
                val_var = tk.DoubleVar(value=addpower)
                entry = tk.Entry(frame, textvariable=val_var, width=6)
                entry.pack(side=tk.LEFT, padx=5)
                self.addpower_vars[addpower] = {"color": color_var, "value": val_var, "button": btn}
        except Exception as e:
            messagebox.showerror("错误", f"生成加光列表失败: {e}")
    
    def run_measurement(self):
        if not self.filename:
            messagebox.showerror("错误", "请先选择CSV文件")
            return
        
        try:
            data = read_data(self.filename)
        except Exception as e:
            messagebox.showerror("错误", f"读取数据失败: {e}")
            return
        
        try:
            ring_choice = self.ring_choice.get()
            center_x = float(self.i_center_x.get())
            center_y = float(self.i_center_y.get())
            # 使用间隔的一半作为测量容差
            measure_threshold = float(self.interval_var.get()) / 2
            diameter = float(self.diameter.get())
        except Exception as e:
            messagebox.showerror("错误", f"参数错误: {e}")
            return
        
        # 构建加光颜色与数值对应关系
        color_value_dict = {}
        for addpower, vars in self.addpower_vars.items():
            color = vars["color"].get()
            value = float(vars["value"].get())
            color_value_dict[color] = value
        
        ring_number_list_input = self.ring_number_list_str.get()
        try:
            ring_number_list = parse_number_range(ring_number_list_input)
        except Exception as e:
            messagebox.showerror("错误", f"微透镜列表解析错误: {e}")
            return
        
        point_per_mm = data.shape[0] / 17
        mm_per_point = 1 / point_per_mm
        semi_diameter = diameter / 2 * point_per_mm
        
        if ring_choice == '内圈':
            image_center = (17/2 * point_per_mm, 17/2 * point_per_mm)
            ring_num = 4
        elif ring_choice == '中圈':
            image_center = ((17/2 + 11) * point_per_mm, 17/2 * point_per_mm)
            ring_num = 7
        elif ring_choice == '外圈':
            image_center = ((17/2 + 16) * point_per_mm, 17/2 * point_per_mm)
            ring_num = 7
        else:
            messagebox.showerror("错误", "无效的圈选")
            return
        
        try:
            binary_image = detect_edge(data, threshold=1.0)
            microlenses, microlens_only_image = label_microlens(binary_image, min_area=15*15)
            microlens_params = microlens_centers_radius(microlenses)
            sorted_microlens_params = cluster_rings(
                microlens_params,
                image_center=image_center,
                ring_num=ring_num,
                max_ring=ring_num+2,
                threshold=10
            )
            sorted_microlens_params = calculate_each_lens_Rx(data, sorted_microlens_params)
            report = measure_list_of_microlens(ring_number_list, point_per_mm, sorted_microlens_params, data, 0.0)
        except Exception as e:
            messagebox.showerror("错误", f"数据处理失败: {e}")
            return
        
        self.report_text.delete(1.0, tk.END)
        self.report_text.insert(tk.END, report)
        
        try:
            power_color_dict = { float(value): color for color, value in color_value_dict.items() }
            fig, checked_microlens = report_checked_microlens(
                sorted_microlens_params, data, power_color_dict,
                radius=semi_diameter, dpi=75, Rx=0.0, threshold=measure_threshold
            )
        except Exception as e:
            messagebox.showerror("错误", f"生成图形失败: {e}")
            return
        
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(fig, master=self.figure_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 清空之前的警告显示
        for widget in self.warning_frame.winfo_children():
            widget.destroy()
        
        # 对于警告微透镜，生成对应图形并嵌入到右侧警告区域
        warning_ids = [i for i, microlens in enumerate(checked_microlens) if microlens["color"] == "warning"]
        for wid in warning_ids:
            try:
                one_lens_fig = report_one_microlens(
                    wid, sorted_microlens_params, data,
                    radius=semi_diameter, mm_per_point=mm_per_point,
                    N_line=6, N_point=100
                )
                self.show_warning_figure(wid, one_lens_fig)
            except Exception as e:
                print(f"绘制微透镜{wid}失败: {e}")
    
    def show_warning_figure(self, wid, fig):
        # 在警告区域内嵌入一个图形，并加上标签
        container = tk.Frame(self.warning_frame, bd=2, relief=tk.GROOVE)
        container.pack(fill=tk.BOTH, expand=True, pady=5)
        label = tk.Label(container, text=f"警告微透镜 {wid}")
        label.pack()
        canvas = FigureCanvasTkAgg(fig, master=container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def on_closing(self):
        plt.close('all')
        self.destroy()
        sys.exit()

if __name__ == "__main__":
    app = MicrolensMeasurementApp()
    app.mainloop()
