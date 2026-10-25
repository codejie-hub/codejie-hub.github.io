# frozen_string_literal: true

module CodeJie
  class LegacyCategories < Jekyll::Generator
    safe true
    priority :low

    def generate(site)
      {
        'vla' => ['VLA', ['机器人学习与 VLA']],
        '控制' => ['控制', ['运动学与控制']],
        '多模态模型' => ['多模态模型', ['大模型基础', '三维视觉', '生成模型']]
      }.each do |slug, (name, topics)|
        page = Jekyll::PageWithoutAFile.new(site, site.source, "categories/#{slug}", 'index.html')
        page.data.merge!('layout' => 'legacy-category', 'title' => "#{name}：分类已整理", 'legacy_topics' => topics)
        site.pages << page
      end
    end
  end
end
