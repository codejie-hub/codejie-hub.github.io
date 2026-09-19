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

本项目使用 Jekyll 和 Chirpy 7.x。Chirpy 要求 Ruby `>= 3.1, < 4.0`，系统自带的 Ruby 3.0.2 无法安装主题；本地使用 Ruby 3.3，与 GitHub Actions 的版本保持一致。

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
