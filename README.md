# Chirpy Starter

[![Gem Version](https://img.shields.io/gem/v/jekyll-theme-chirpy)][gem]&nbsp;
[![GitHub license](https://img.shields.io/github/license/cotes2020/chirpy-starter.svg?color=blue)][mit]

When installing the [**Chirpy**][chirpy] theme through [RubyGems.org][gem], Jekyll can only read files in the folders
`_data`, `_layouts`, `_includes`, `_sass` and `assets`, as well as a small part of options of the `_config.yml` file
from the theme's gem. If you have ever installed this theme gem, you can use the command
`bundle info --path jekyll-theme-chirpy` to locate these files.

The Jekyll team claims that this is to leave the ball in the user’s court, but this also results in users not being
able to enjoy the out-of-the-box experience when using feature-rich themes.

To fully use all the features of **Chirpy**, you need to copy the other critical files from the theme's gem to your
Jekyll site. The following is a list of targets:

```shell
.
├── _config.yml
├── _plugins
├── _tabs
└── index.html
```

To save you time, and also in case you lose some files while copying, we extract those files/configurations of the
latest version of the **Chirpy** theme and the [CD][CD] workflow to here, so that you can start writing in minutes.

## Usage

Check out the [theme's docs](https://github.com/cotes2020/jekyll-theme-chirpy/wiki).

## 本地预览

本项目使用 Jekyll 和固定版本 Chirpy 7.3.1，提交 Gemfile.lock 以保持本地与 CI 一致。Chirpy 要求 Ruby `>= 3.1, < 4.0`，系统自带的 Ruby 3.0.2 无法安装主题；本地使用 Ruby 3.3，与 GitHub Actions 的版本保持一致。

如果已安装 Conda，可以创建独立环境。编译工具用于安装含原生扩展的 Ruby gems：

```bash
conda create -n blog-preview --override-channels -c conda-forge \
  ruby=3.3 c-compiler cxx-compiler make pkg-config
conda activate blog-preview

ruby --version
bundle --version
```

在仓库根目录安装依赖并启动：

```bash
bundle config set --local path vendor/bundle
bundle install
bundle exec jekyll serve --livereload --future
```

打开 <http://localhost:4001/>。例如，强化学习基础文章的地址为 <http://localhost:4001/posts/rl-base/>。保存文章后会自动重新构建并刷新页面；`--future` 允许预览日期晚于本机时间的文章。修改 `_config.yml` 后需要重启服务，按 `Ctrl+C` 可停止服务。

以后只需在仓库根目录执行：

```bash
conda activate blog-preview
bundle exec jekyll serve --livereload --future
```

Jekyll 预览使用两个独立端口：本项目在 `_config.yml` 中将网页端口设为 `4001`，避开本机已被占用的 Jekyll 默认端口 `4000`；LiveReload 自动刷新默认是 `35729`。如果已有预览在运行，直接使用它，或先在原终端按 `Ctrl+C` 停止，再重新启动。

如果提示网页端口 `Address already in use`，可以只更换网页端口：

```bash
bundle exec jekyll serve --livereload --future --port 4002
```

此时首页为 <http://localhost:4002/>。

如果错误来自 `live_reload_reactor.rb`，并提示 `no acceptor (port is in use or requires root privileges)`，需要检查 LiveReload 端口；仅修改 `--port` 不会修改它。确需运行另一个预览时，为两者分别指定空闲端口：

```bash
bundle exec jekyll serve --livereload --future --port 4002 --livereload-port 35730
```

Linux 上可以用以下命令查看占用情况，确认进程后再决定停止哪个预览：

```bash
ss -ltnp '( sport = :4000 or sport = :4001 or sport = :4002 or sport = :35729 or sport = :35730 )'
```

## Contributing

This repository is automatically updated with new releases from the theme repository. If you encounter any issues or want to contribute to its improvement, please visit the [theme repository][chirpy] to provide feedback.

## License

This work is published under [MIT][mit] License.

[gem]: https://rubygems.org/gems/jekyll-theme-chirpy
[chirpy]: https://github.com/cotes2020/jekyll-theme-chirpy/
[CD]: https://en.wikipedia.org/wiki/Continuous_deployment
[mit]: https://github.com/cotes2020/chirpy-starter/blob/master/LICENSE

## 内容与插图维护

主分类来自 `_data/domains.yml`，每篇只设置一个 `categories`，并填写 `content_type: tutorial / paper / survey / qa`。博客展示全集，论文页展示 paper 与 survey。

正文图片统一使用 `_includes/figure.html`，交互演示使用 `_includes/demo-frame.html`。SVG 为可编辑图源；不要用封面代替方法图。

```powershell
python tools/build-illustrations.py
python tools/check-content.py
bundle exec jekyll build
bundle exec htmlproofer _site --disable-external
```

图文与分类清单见 `reports/blog-organization.md`。`tools/organize-content.py` 为本次幂等迁移脚本；新增文章通常只需编辑元数据与正文，无需重跑迁移。

Windows 的 HTMLProofer 若找不到 libcurl，可复用 Git for Windows 提供的 DLL；以下设置只影响当前终端。将路径替换为自己的 Git 安装位置：

```powershell
$env:DEBUG=$null
$env:BLOG_CURL_DLL='D:/Git/mingw64/bin/libcurl-4.dll'
$env:RUBYOPT='-r./tools/windows-curl.rb'
bundle exec htmlproofer _site --disable-external
```

GitHub Actions 的 Linux 环境使用普通 HTMLProofer 命令即可。

## 具身智能学习系列

2026-10-24 新增五篇基础教程，阅读顺序记录在 `_data/embodied_series.yml`，每篇顶部提供完整系列导航：模仿学习 → 三维抓取几何 → 接触与阻抗控制 → 世界模型与规划 → 离线强化学习。五篇均使用现有主题分类；博客合计 23 篇，论文页仍为 9 篇。目录与验证记录见 [学习系列交付记录](reports/embodied-learning.md)。

```powershell
python tools/build-embodied-figures.py
python tools/check-embodied-examples.py  # 运行正文中的五个完整例子，需要 NumPy
python tools/check-content.py
bundle exec jekyll build --destination .qa/site
python tools/preview-server.py --directory .qa/site --port 4174
```

预览服务将 SVG 明确设为 `image/svg+xml`，避免 Windows 注册表中的错误类型导致浏览器下载或空白显示。在另一个终端检查五篇文章的三种宽度和明暗主题：

```powershell
node tools/browser-audit.mjs series http://127.0.0.1:4174
```

封面 JPEG 由 SVG 图源渲染。若修改新系列图源，先运行生成脚本，再用 `python tools/preview-server.py --directory . --port 4173` 服务仓库根目录；另一个终端运行 `node tools/browser-audit.mjs series-assets http://127.0.0.1:4173`，之后重新构建站点。该模式只更新本系列封面，图源和原有图片均保留。

`check-content.py --baseline` 针对上一轮 18 篇的历史验收快照；新增文章后使用默认检查。两轮验收记录分别保留，避免把历史数量当作当前整站数量。
