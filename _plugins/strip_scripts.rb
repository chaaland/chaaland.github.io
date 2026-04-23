module Jekyll
  module StripScriptsFilter
    def strip_scripts(input)
      input.gsub(/<script\b[^>]*>.*?<\/script>/mi, '')
    end
  end
end

Liquid::Template.register_filter(Jekyll::StripScriptsFilter)